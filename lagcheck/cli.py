import argparse
import csv
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import sys
from . import __version__
from .models import InputError, IntegrityError
from .ms import load_signal
from .laser import load_laser, export_corrected, _time
from .estimator import diagnose
from .reference_core import MIN_LINES_PER_BLOCK
from .verification import export_plan, verify_written_log
from .fallback import try_alignment_fallback
from .signal_selection import MODES as SELECTION_MODES, select_signal
from .eligibility import assess as assess_eligibility
from .tolerance import read_request, resolve as resolve_tolerance, spot_width, distance_scale


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def write_blocks(out, blocks, prefix='blocks'):
    write_json(out/(prefix+'.json'), blocks)
    if blocks:
        keys = sorted(set().union(*(r.keys() for r in blocks)))
        with (out/(prefix+'.csv')).open('x', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
                         for k,v in row.items()} for row in blocks)


def _reject_known_corrected_input(path):
    input_hash = digest(path)
    for name in ('correction_manifest.json', 'provisional_manifest.json'):
        manifest = path.parent/name
        if not manifest.is_file():
            continue
        data = json.loads(manifest.read_text(encoding='utf-8'))
        if data.get('output_sha256') == input_hash:
            raise InputError('This file matches an earlier correction manifest (verified or provisional). Use the original laser log for export; '
                             'diagnosis without --export remains available. Do not apply an offset or rate twice.')


_EXPORT_KEYS = {'mode', 'ms_first_sample_at_laser_time', 'reference_note', 'provisional', 'reference_rate_ppm'}


def _validate_reference_rate(value):
    if value is None:
        return
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or 1 + value * 1e-6 <= 0):
        raise InputError('export.reference_rate_ppm must be a finite number in µs s⁻¹ giving a positive time-scale factor')


def _validate_export_options(options, laser):
    """Reject export configuration errors before any output is written (exit code 1)."""
    unknown = set(options) - _EXPORT_KEYS
    if unknown:
        raise InputError('Unknown export setting(s): ' + ', '.join(sorted(unknown)))
    mode = options.get('mode', 'rate_only')
    if mode not in ('rate_only', 'affine'):
        raise InputError('export.mode must be rate_only or affine')
    if not isinstance(options.get('provisional', True), bool):
        raise InputError('export.provisional must be true or false')
    _validate_reference_rate(options.get('reference_rate_ppm'))
    if mode == 'rate_only':
        if 'ms_first_sample_at_laser_time' in options:
            raise InputError('An explicit time reference requires export.mode=affine')
        return
    reference, note = options.get('ms_first_sample_at_laser_time'), options.get('reference_note')
    if not isinstance(reference, str) or not reference.strip() or not isinstance(note, str) or not note.strip():
        raise InputError('Affine export needs ms_first_sample_at_laser_time and a reference_note explaining its source; no clock origin is guessed')
    _, template = _time(reference, laser.metadata['config'])
    original_template = laser.table['templates'][0][0]
    if template is not None and ((template.tzinfo is None) != (original_template.tzinfo is None)):
        raise InputError('The time reference and laser timestamps must have consistent timezone awareness')


def _check_written_bytes(receipt, path):
    """The written file must still be the bytes the exporter checked."""
    if digest(path) != receipt['output_sha256']:
        raise InputError('Written file differs from the checked output bytes')


def _check_written_plan(receipt, rate, plan):
    """The written file must apply exactly the planned rate and constant."""
    if abs(float(receipt['rate_ppm']) - float(rate)) > 1e-9 * max(1.0, abs(float(rate))):
        raise InputError('Written rate differs from the export plan')
    if Decimal(str(receipt['offset_seconds'])) != Decimal(str(plan['offset_seconds'])):
        raise InputError('Written constant differs from the export plan')


def _check_written_mapping(receipt, rate, plan, path):
    """The written file must be the checked bytes and apply exactly the planned rate and constant."""
    _check_written_bytes(receipt, path)
    _check_written_plan(receipt, rate, plan)


QUALIFIED_STATUS = 'supported_excluding_record_boundary_blocks'


def _screen_summary(report):
    """What the laser-log screen found, kept with every written log so that a changed limit stays visible."""
    screen = report.get('laser_log_screen') or {}
    return dict(evaluated=screen.get('evaluated'), limit_s=screen.get('limit_s'), default_limit_s=screen.get('default_limit_s'),
                flagged_intervals=len(screen.get('flagged') or []),
                n_flagged_at_default_limit=screen.get('n_flagged_at_default_limit'))


# Block counts of the stability alternatives in estimator._diagnose_standard. With the baseline block
# count they set how many firing intervals the complete analysis needs (MIN_LINES_PER_BLOCK per block).
STABILITY_BLOCK_COUNTS = (8, 12, 15)
# Criteria of the automatic starting-alignment search in estimator._coarse.
ALIGNMENT_CRITERIA = 'An identifiable start needs a peak above 0.25, a margin above 0.01 and a peak inside the search window.'
EXPORT_HINT = ('To write one, run again with --export and a new output directory. A verified log is written only when '
               'the status is supported and the read-back of the written file passes. Otherwise a provisional (not verified) '
               'log is written when a rate is available and eligible for provisional export (see provisional_export_eligibility), '
               'unless --no-provisional is given.')
# Ends the note and the manifest instruction of a provisional log whose eligibility record holds warnings,
# so that a reader of those texts alone is led to them. A log without warnings keeps its texts as they were.
WARNINGS_POINTER = ' See also the warnings recorded in provisional_export_eligibility.warnings.'
# The texts below explain a completed result. They read numbers already in the report and must not
# turn a completed diagnosis into an error, so an incomplete report simply gets less text.
_TEXT_ERRORS = (KeyError, IndexError, TypeError, ValueError, AttributeError)


def _against(measured, limit):
    """Two numbers as text, with as few significant digits as still tell them apart."""
    for digits in range(4, 18):
        a, b = f'{measured:.{digits}g}', f'{limit:.{digits}g}'
        if a != b:
            break
    return a, b


def _rate_summary(report):
    """Which rate is reported, and the single-pass rate with the standard error that belongs to it.

    The standard error in the report is that of the slope of the single-pass fit. It is not an
    uncertainty of the rate after iteration, so that rate is stated without one.
    """
    empty = dict(rate_kind=None, single_pass_rate_ppm=None, single_pass_rate_se_ppm=None, single_pass_blocks=None)
    try:
        baseline, closure = report.get('baseline') or {}, report.get('closure') or {}
        if report.get('rate_ppm') is None or baseline.get('rate_ppm') is None:
            return empty
        return dict(rate_kind='iterated' if closure.get('converged') else 'single_pass',
                    single_pass_rate_ppm=baseline['rate_ppm'], single_pass_rate_se_ppm=baseline.get('slope_se_ppm'),
                    single_pass_blocks=baseline.get('n_retained'))
    except _TEXT_ERRORS:
        return empty


def _rate_lines(report):
    summary = _rate_summary(report)
    if summary['rate_kind'] is None:
        return []
    kind = ('rate after iteration, reported without a standard error' if summary['rate_kind'] == 'iterated' else
            'single-pass rate; the iteration did not converge')
    return [f"Kind of rate estimate: {kind}",
            f"Single-pass rate (µs s⁻¹): {summary['single_pass_rate_ppm']}; standard error of its slope (µs s⁻¹): "
            f"{summary['single_pass_rate_se_ppm']} ({summary['single_pass_blocks']} blocks)"]


def _lines_in_fits(report):
    edge = report.get('record_edge_lines') or {}
    return report['n_intervals'] - (edge.get('skipped_at_start') or 0) - (edge.get('skipped_at_end') or 0)


def _iteration_state(report):
    """Where the iteration ended: the last trial rate, the residual rate left there and the convergence limit."""
    closure, limit = report['closure'], report['settings']['closure_tolerance_ppm']
    last, steps = closure['history'][-1], len(closure['history'])
    if last['residual_ppm'] is None:
        return f"No fit was obtained at the trial rate of iteration {steps} ({last['trial_ppm']:.3f} µs s⁻¹)."
    a, b = _against(abs(last['residual_ppm']), limit)
    return (f"After {steps} iteration(s) the trial rate was {last['trial_ppm']:.3f} µs s⁻¹ and the residual rate "
            f"{'-' if last['residual_ppm'] < 0 else '+'}{a} µs s⁻¹; convergence needs an absolute residual below {b} µs s⁻¹.")


def _off_line(numbers, rows, fit, tol):
    """Blocks whose lag disagrees with the fitted line, with the difference where the block has one midpoint value."""
    by_number = {int(b['block']): b for b in rows}
    parts = []
    for number in numbers:
        b = by_number.get(int(number))
        if b is None:
            parts.append(f'block {number}')
        elif (b.get('fit_eligible') or b.get('record_boundary_excluded')) and b.get('lag_s') is not None:
            difference = b['lag_s'] - (fit['intercept_s'] + fit['rate_ppm']*1e-6*(b['t_mid'] - fit['intercept_time_s']))
            parts.append(f"block {number} ({'-' if difference < 0 else '+'}{_against(abs(difference), tol)[0]} s)")
        else:
            parts.append(f'block {number} (clipped by the signal record)')
    return [f"Lag differs from the fitted line by more than the tolerance of {tol:g} s: {', '.join(parts)}."] if parts else []


def _at_limit(rows, retained):
    parts = [f"block {int(b['block'])} (lag {b['lag_s']:.4g} s, search range {b['search_min_s']:.4g} to {b['search_max_s']:.4g} s)"
             for b in rows if int(b['block']) in retained and b.get('near_search_boundary')]
    return [f"Correlation peak within two sampling intervals of a search limit: {', '.join(parts)}."] if parts else []


def _no_rate_reason(variant, report):
    config = variant['config']
    if config == 'closure':
        return 'the iteration did not converge'
    if config.startswith('n_blocks='):
        needed, used = int(config.split('=')[1])*MIN_LINES_PER_BLOCK, _lines_in_fits(report)
        if used < needed:
            return f'needs {needed} firing intervals; {used} are in the fits'
    return 'no rate could be fitted with this setting'


def _variant_failure(variant, report):
    if variant['rate_ppm'] is None:
        return _no_rate_reason(variant, report)
    reasons = []
    if variant.get('coverage_required') and variant['coverage_fraction'] < report['settings']['min_coverage']:
        reasons.append('coverage fraction ' + _against(variant['coverage_fraction'], report['settings']['min_coverage'])[0])
    if variant.get('peak_at_search_boundary'):
        reasons.append('correlation peak at a search limit')
    if variant.get('contradictory_high_quality_blocks'):
        reasons.append('block(s) ' + ', '.join(str(b) for b in variant['contradictory_high_quality_blocks']) + ' off the fitted line')
    # An alternative passes on coverage, search limit, agreement and held-out prediction. Its held-out
    # error is not stored, so that condition is named when the other three are met.
    return ', '.join(reasons) or 'held-out prediction error above the tolerance'


def _check_detail(check, report, blocks):
    """Measured value and limit behind a check that was not met, from numbers already in the report."""
    try:
        s, closure, tol = report['settings'], report.get('closure') or {}, report['settings']['tolerance_s']
        if check.startswith('closure_') and check != 'closure_converged' and not closure.get('residual_fit'):
            return ['No fit was obtained in the last iteration.']
        if check == 'coverage':
            a, b = _against(report['coverage_fraction'], s['min_coverage'])
            return [f'Coverage fraction {a}; at least {b} is required.']
        if check == 'closure_coverage':
            a, b = _against(closure['coverage_fraction'], s['min_coverage'])
            return [f'Coverage fraction after the last iteration {a}; at least {b} is required.']
        if check == 'no_contradictory_high_quality_blocks':
            return _off_line(report['contradictory_high_quality_blocks'], blocks or [], report['baseline'], tol)
        if check == 'closure_no_contradictory_high_quality_blocks':
            return _off_line(closure['contradictory_high_quality_blocks'], closure['blocks'], closure['residual_fit'], tol)
        if check == 'peak_not_at_search_boundary':
            return _at_limit(blocks or [], report['baseline']['retained'])
        if check == 'closure_peak_not_at_boundary':
            return _at_limit(closure['blocks'], closure['retained_blocks'])
        if check in ('baseline_heldout', 'closure_heldout'):
            fit, stage = (report['baseline'], 'of the single-pass fit') if check == 'baseline_heldout' else \
                (closure['residual_fit'], 'in the last iteration')
            a, b = _against(fit['heldout_max_s'], tol)
            return [f'Largest held-out prediction error {stage}: {a} s; the tolerance is {b} s.']
        if check == 'all_sensitivity_configs_estimable':
            return [f"{v['config']}: {_no_rate_reason(v, report)}" for v in report['sensitivity'] if v['rate_ppm'] is None]
        if check == 'all_sensitivity_evidence_supported':
            return [f"{v['config']}: {_variant_failure(v, report)}" for v in report['sensitivity'] if not v['evidence_pass']]
        if check == 'sensitivity_envelope':
            rates = [report['baseline']['rate_ppm']] + [v['rate_ppm'] for v in report['sensitivity'] if v['rate_ppm'] is not None]
            a, b = _against(report['sensitivity_envelope_s'], tol)
            return [f"Rates of the single-pass fit and the alternatives range from {min(rates):.3f} to {max(rates):.3f} µs s⁻¹. "
                    f"Over the {report['duration_s']:.0f} s from the first firing to the last, this spread is {a} s; the tolerance is {b} s."]
        if check == 'closure_converged':
            return [_iteration_state(report)]
        if check == 'closure_residual':
            a, b = _against(abs(closure['residual_ppm'])*1e-6*report['duration_s'], tol)
            return [f"A residual rate of {closure['residual_ppm']:.4g} µs s⁻¹ over {report['duration_s']:.0f} s amounts to {a} s; "
                    f"the tolerance is {b} s."]
        if check == 'no_flagged_line_change_interval':
            return [f"{len(report['laser_log_screen']['flagged'])} line-change interval(s) flagged; "
                    'they are listed under the laser-log screen below.']
        if check == 'initial_alignment_identifiable':
            return [f"Starting-alignment search: peak {report['alignment']['peak']:.4g}, margin {report['alignment']['margin']:.4g}. "
                    + ALIGNMENT_CRITERIA]
    except _TEXT_ERRORS:
        pass
    return []


def _readback_detail(check, post, report):
    """Measured value and limit behind a read-back check that was not met, from numbers in post_export_verification."""
    try:
        tol, rates = post['tolerance_s'], (post['residual_baseline_rate_ppm'], post['residual_rate_ppm'])
        if check == 'full_diagnostic_supported':
            return ['The diagnostic of the written log did not reach the required status; post_export_report.json holds its status and checks.']
        if check in ('residual_rate_below_threshold', 'residual_drift_within_tolerance'):
            if None in rates:
                return ['The diagnostic of the written log gave no residual rate.']
            largest = max(abs(x) for x in rates)
            if check == 'residual_rate_below_threshold':
                a, b = _against(largest, post['residual_rate_limit_ppm'])
                return [f'Larger of the two residual rates in absolute value: {a} µs s⁻¹; it must be below {b} µs s⁻¹.']
            return [f'The larger residual rate, {largest:.4g} µs s⁻¹ in absolute value, times the duration of the written log '
                    f'(duration_s in post_export_report.json) exceeds the tolerance of {tol:g} s.']
        if check == 'coverage':
            if post['coverage_fraction'] is None:
                return ['The diagnostic of the written log gave no fit.']
            a, b = _against(post['coverage_fraction'], report['settings']['min_coverage'])
            return [f'Coverage fraction {a}; at least {b} is required.']
        if check in ('heldout', 'fixed_reference_alignment_within_tolerance'):
            key, name = (('heldout_max_s', 'Largest held-out prediction error') if check == 'heldout' else
                         ('fixed_reference_max_lag_s', 'Largest absolute lag of a retained block under the fixed mapping'))
            if post[key] is None:
                return ['The diagnostic of the written log gave no fit.']
            a, b = _against(post[key], tol)
            return [f'{name}: {a} s; the tolerance is {b} s.']
    except _TEXT_ERRORS:
        pass
    return []


def _stopped_before_checks(report):
    """Why the diagnosis stopped before any check, with the numbers recorded for that reason."""
    reason = str(report.get('reason') or '')
    lines = [f"No check was evaluated: the diagnosis stopped before the checks ({reason or 'no reason recorded'})."]
    try:
        s = report['settings']
        if reason.startswith('time_grid_not_supported'):
            a, b = _against(report['max_step_relative_deviation'], s['max_step_relative_deviation'])
            lines.append(f'Largest relative deviation of a sampling interval from the mean spacing: {a}; '
                         f'the limit max_step_relative_deviation is {b}.')
        elif reason.startswith('index_grid_approximation'):
            a, b = _against(report['index_lag_error_bound_s'], s['tolerance_s']/4)
            lines.append(f'Bound on the lag error from treating the samples as equally spaced: {a} s; '
                         f'the limit is a quarter of the tolerance, {b} s.')
        elif reason.startswith('insufficient_laser_intervals'):
            lines.append(f"{report['n_intervals']} firing intervals; {s['n_blocks']} blocks of at least {MIN_LINES_PER_BLOCK} "
                         f"need {s['n_blocks']*MIN_LINES_PER_BLOCK}.")
        elif reason.startswith('initial_alignment_not_identifiable'):
            found = report.get('alignment') or {}
            if found.get('reason') == 'no_temporal_contrast':
                lines.append('The firing pattern or the matching signal shows no contrast over time.')
            elif found.get('reason') == 'coarse_window_too_small':
                lines.append(f"The starting-alignment window (coarse_window_s = {s['coarse_window_s']:g} s) is too small for a search "
                             f"at a mean sample spacing of {report['mean_spacing_s']:.4g} s.")
            else:
                lines.append(f"Starting-alignment search: peak {found['peak']:.4g}, margin {found['margin']:.4g}, "
                             f"peak at the window limit: {'yes' if found['boundary'] else 'no'}. " + ALIGNMENT_CRITERIA)
        elif reason.startswith('fewer_than_four_usable_blocks'):
            lines.append(f"{report['gate_passing_blocks']} of {report['computed_blocks']} blocks passed the correlation gates "
                         f"(peak above {s['peak_min']:g}, margin above {s['margin_min']:g}). "
                         'A fit needs at least four such blocks that also lie inside the signal record.')
    except _TEXT_ERRORS:
        pass
    if report.get('computed_blocks'):
        lines.append('blocks.csv lists the lag, correlation peak, margin and reasons of every block.')
    return lines


def _record_length_note(report):
    """Say so when the record has too few firing intervals for a fit or for every stability alternative."""
    n, blocks = report['n_intervals'], report['settings']['n_blocks']
    if str(report.get('reason') or '').startswith('insufficient_laser_intervals') and report.get('rate_ppm') is None:
        return (f'Record too short: {n} firing intervals. A fit with {blocks} blocks needs at least {blocks*MIN_LINES_PER_BLOCK} '
                f'({MIN_LINES_PER_BLOCK} per block), so no rate was estimated and no check was evaluated.')
    used, largest = _lines_in_fits(report), max(STABILITY_BLOCK_COUNTS + (blocks,))
    if used >= largest*MIN_LINES_PER_BLOCK:
        return None
    missing = [f'n_blocks={k}' for k in STABILITY_BLOCK_COUNTS if used < k*MIN_LINES_PER_BLOCK]
    note = (f'Record too short for the complete analysis: {n} firing intervals'
            + (f', of which {used} are in the fits after the lines at the ends of the record were left out' if used != n else '')
            + f'. The complete analysis needs {largest*MIN_LINES_PER_BLOCK} ({largest} blocks of at least {MIN_LINES_PER_BLOCK}). '
            + f"The stability alternative(s) {', '.join(missing)} cannot be estimated.")
    if 'all_sensitivity_configs_estimable' in (report.get('checks') or {}):
        note += (' The checks all_sensitivity_configs_estimable and all_sensitivity_evidence_supported need every '
                 'alternative and are therefore reported as not met.')
    if report.get('short_acquisition'):
        note += ' See the conditional short-acquisition analysis in this report.'
    return note


def _iteration_note(report):
    """Say which rate is reported when the iteration did not converge."""
    closure, baseline = report.get('closure') or {}, report.get('baseline') or {}
    if not closure.get('history') or closure.get('converged') or baseline.get('rate_ppm') is None:
        return None
    return ('The iteration did not converge. ' + _iteration_state(report) + ' The reported rate estimate is the single-pass rate '
            f"({baseline['rate_ppm']:.3f} µs s⁻¹), not the last trial rate.")


def _selected_diagnostic(report):
    """The diagnostic an export uses, with its rate source: the adopted input-placement retry, otherwise the main diagnostic."""
    fallback = report.get('alignment_fallback') or {}
    if fallback.get('adopted'):
        return fallback['diagnostic'], 'input_position_fallback'
    return report, 'acquisition_estimate'


def _withheld_message(eligibility, reference_used=False, view_available=False):
    """Why no provisional log was written from the estimate. The view is named only once it has been drawn."""
    what = ('the estimate at the original input placement' if eligibility['candidate']['source'] == 'input_position_fallback'
            else 'this acquisition estimate')
    return (f"No provisional log was written from {what}: {'; '.join(eligibility['reasons']) or eligibility['status']}. "
            'The estimate and failed checks remain in the report.'
            + (' An uncorrected inspection view is available.' if view_available else '')
            + (' The provisional log applies the supplied reference rate instead.' if reference_used else
               ' A documented reference rate may be supplied.'))


def _applied_mapping(report, eligibility, reference_rate_ppm):
    """The mapping that the written log applies, or that a preview displays when no log was written.

    The log and the preview read this one record, so a rate that is withheld
    from the provisional log is not drawn as an applied correction. A supported
    diagnosis is exported through the verified path and keeps its estimate.
    """
    fit_report, source = _selected_diagnostic(report)
    estimate = fit_report.get('rate_ppm')
    usable = estimate is not None and (report['status'] == 'supported' or eligibility['status'] == 'not_blocked')
    rejected = None if estimate is None or usable else dict(rate_ppm=estimate, rate_source=source,
                                                            label='rejected candidate for inspection')
    if report.get('usable_file'):
        return dict(kind='corrected_laser_log', rate_ppm=report['rate_ppm'], rate_source='acquisition_estimate',
                    file=report['usable_file'], rejected_candidate=None)
    if report.get('provisional_file'):
        applied = report.get('provisional_rate_source')
        return dict(kind='provisional_laser_log', rate_ppm=report.get('provisional_rate_ppm'), rate_source=applied,
                    file=report['provisional_file'], rejected_candidate=rejected if applied == 'user_reference_rate' else None)
    if usable:
        return dict(kind='display_only', rate_ppm=estimate, rate_source=source, file=None, rejected_candidate=None)
    if reference_rate_ppm is not None:
        return dict(kind='display_only', rate_ppm=float(reference_rate_ppm), rate_source='user_reference_rate', file=None,
                    rejected_candidate=rejected)
    return dict(kind='none', rate_ppm=None, rate_source=None, file=None, rejected_candidate=rejected)


def _reference_rate_use(report, reference_rate_ppm, export, provisional):
    """Whether a supplied reference rate was applied to a log, and the reason when it was not."""
    fallback = report.get('alignment_fallback') or {}
    estimate = (fallback['diagnostic'] if fallback.get('adopted') else report).get('rate_ppm')
    eligibility = report.get('provisional_export_eligibility') or {}
    # An estimate that is withheld from the provisional log does not take precedence over the reference rate.
    ineligible = (estimate is not None and report.get('status') != 'supported'
                  and eligibility.get('status') in ('withheld', 'not_assessable'))
    start = f'The reference rate ({float(reference_rate_ppm):g} µs s⁻¹) was '
    if report.get('provisional_rate_source') == 'user_reference_rate':
        if ineligible:
            return dict(supplied_ppm=reference_rate_ppm, used=True, reason='acquisition_estimate_not_eligible',
                        note=start + f'applied to the provisional log because the rate estimated from this acquisition '
                        f'({estimate:.3f} µs s⁻¹) was not eligible for provisional export. That estimate remains in the report.')
        return dict(supplied_ppm=reference_rate_ppm, used=True, reason='no_acquisition_estimate',
                    note=start + 'applied to the provisional log because no rate could be estimated from this acquisition.')
    if estimate is not None and not ineligible:
        origin = ' at the original input placement' if fallback.get('adopted') else ''
        return dict(supplied_ppm=reference_rate_ppm, used=False, reason='acquisition_estimate_available',
                    note=start + f'not used: a rate was estimated from this acquisition{origin} ({estimate:.3f} µs s⁻¹), and that '
                    'estimate takes precedence. A reference rate is applied only when the acquisition gives no estimate or its estimate is not eligible for provisional export.')
    if not export:
        return dict(supplied_ppm=reference_rate_ppm, used=False, reason='export_not_requested',
                    note=start + 'not applied to a log because --export was not requested. '
                    + ('The rate estimated from this acquisition is not eligible for provisional export, ' if ineligible else
                       'No rate could be estimated from this acquisition, ')
                    + 'so a preview of this run, if one is drawn, displays the reference rate.')
    if not provisional:
        return dict(supplied_ppm=reference_rate_ppm, used=False, reason='provisional_output_disabled',
                    note=start + 'not used: provisional output is disabled, and a reference rate is applied only to a provisional log.')
    return dict(supplied_ppm=reference_rate_ppm, used=False, reason='not_applied', note=start + 'not applied to a log.')


def _annotate(report, reference_rate_ppm, export, provisional):
    """Add plain statements about this result for report.md and the command line; no existing field is changed."""
    notes = []
    for build in (_record_length_note, _iteration_note):
        try:
            note = build(report)
        except _TEXT_ERRORS:
            note = None
        if note:
            notes.append(note)
    if (report.get('provisional_export_eligibility') or {}).get('message'):
        notes.append(report['provisional_export_eligibility']['message'])
    if reference_rate_ppm is not None:
        try:
            report['reference_rate'] = _reference_rate_use(report, reference_rate_ppm, export, provisional)
            notes.append(report['reference_rate']['note'])
        except _TEXT_ERRORS:
            pass
    report['notes'] = notes


def _distance_lines(report):
    """The tolerance of the run and, where a scan speed is known, time differences as distances along the scan."""
    resolution, scale = report.get('tolerance_resolution'), report.get('distance_scale')
    if not isinstance(resolution, dict) or not isinstance(scale, dict):
        return []
    lines = ['', '## Tolerance and equivalent distances', '']
    try:
        if resolution['requested_unit'] == 'um':
            lines.append(f"Tolerance requested: {resolution['requested_value']:g} µm ({resolution['request_source']}). It was converted "
                         f"once, before the diagnosis, to {resolution['resolved_tolerance_s']} s at {resolution['speed_um_s']:g} µm s⁻¹, "
                         'the largest scan speed of the valid firing intervals. The checks and the read-back use this value in seconds.')
        else:
            lines.append(f"Tolerance: {resolution['resolved_tolerance_s']} s ({resolution['request_source']}).")
        replaced = resolution.get('overridden_configuration_value')
        if replaced:
            lines.append(f"The command line replaced the tolerance of the configuration ({replaced['value']} "
                         f"{'µm' if replaced['unit'] == 'um' else 's'}, {replaced['source']}).")
        if resolution.get('reused_from_saved_configuration'):
            lines.append('This tolerance was resolved by an earlier run and reused from its saved configuration. It was not converted again.')
        speed, spot = scale['speed'], scale['spot_width']
        if scale['status'] != 'computed':
            lines.append(f"Equivalent distances: not computable ({scale['reason']}).")
            return lines
        both = speed['min_um_s'] != speed['max_um_s']
        lines.append('Scan speed of the firing intervals: '
                     + (f"{speed['min_um_s']:g} to {speed['max_um_s']:g}" if both else f"{speed['max_um_s']:g}")
                     + f" µm s⁻¹ ({speed['source']}; {speed['n_firing_intervals']} firing intervals).")
        if speed.get('declared_differs_from_log'):
            lines.append('The declared speed differs from the speed recorded in the laser log ('
                         + ', '.join(f'{v:g}' for v in speed['log_values_um_s']) + ' µm s⁻¹). The declared value is used.')
        lines.append('Spot width along the scan: ' + (f"{spot['width_um']:g} µm ({spot['source']})." if spot['status'] == 'known' else
                                                     f"not computable ({spot['reason']})."))
        for row in scale['quantities']:
            text = f"- {row['quantity']}: {row['seconds']:.4g} s; " + (f"{row['um_min']:.4g} to {row['um_max']:.4g}" if both else
                                                                     f"{row['um_max']:.4g}") + ' µm'
            if row['spot_widths_max'] is not None:
                text += ' (' + (f"{row['spot_widths_min']:.3g} to {row['spot_widths_max']:.3g}" if both else
                                f"{row['spot_widths_max']:.3g}") + ' spot widths)'
            lines.append(text + ('' if row['limit_s'] is None else f"; limit {row['limit_s']:.4g} s") + f" ({row['basis']})")
        lines.extend(['Not converted: ' + '; '.join(scale['not_converted']) + '.', scale['note']])
    except _TEXT_ERRORS:
        pass
    return lines


def _eligibility_lines(report):
    """The provisional export eligibility of the selected estimate, with the numbers behind each condition."""
    record = report.get('provisional_export_eligibility')
    if not isinstance(record, dict):
        return []
    lines = ['', '## Provisional export eligibility', '']
    try:
        lines.extend([f"Status: {record['status']}" + (f"; evidence {record['evidence_completeness']}"
                                                         if record.get('evidence_completeness') else ''), record['explanation']])
        c, tol = record['candidate'], record['tolerance_s']
        if record['status'] != 'no_acquisition_rate':
            kind = 'rate after iteration' if c['rate_kind'] == 'iterated' else 'single-pass rate'
            lines.extend([f"Candidate: {c['source']}, {c['rate_ppm']} µs s⁻¹ ({kind}). Single-pass rate {c['single_pass_rate_ppm']} µs s⁻¹; "
                          f"standard error of its slope {c['single_pass_rate_se_ppm']} µs s⁻¹ ({c['single_pass_blocks']} blocks).",
                          f"Firing span, first firing start to last firing end: {record['duration_s']} s. Tolerance: {tol} s."])
            cover = record.get('fit_coverage') or {}
            if cover.get('coverage_fraction') is not None and cover.get('first_block_mid_s') is not None:
                lines.append(f"Fit coverage: coverage fraction {cover['coverage_fraction']:.4g}; midpoints of the first and last retained block "
                             f"at {cover['first_block_mid_s']:.1f} s and {cover['last_block_mid_s']:.1f} s (t_mid of its blocks); "
                             f"{cover['lines_left_out_at_start'] or 0} and {cover['lines_left_out_at_end'] or 0} firing intervals left out "
                             'of the fits at the start and at the end of the record.')
            d, s, i = (record['conditions'][name] for name in ('rate_direction', 'sign_across_settings', 'unconverged_iteration'))

            def missing(condition):
                return f" (not available: {', '.join(condition['not_available'])})" if condition.get('not_available') else ''
            if d['state'] == 'not_assessable':
                lines.append('- rate_direction: not_assessable' + missing(d))
            else:
                lines.append(f"- rate_direction: {d['state']}. Screening half-width {d['half_width_ppm']:.4g} µs s⁻¹ "
                             f"({d['t_quantile']:.4g} times the standard error); the single-pass rate is "
                             + ('within this half-width of zero' if d['includes_zero'] else 'more than this half-width from zero')
                             + f"; over the firing span the half-width amounts to {_against(d['effect_s'], tol)[0]} s.")
            if s['state'] == 'not_assessable':
                lines.append('- sign_across_settings: not_assessable' + missing(s))
            else:
                lines.append(f"- sign_across_settings: {s['state']}. The single-pass rate, the candidate rate and the rates of the "
                             f"alternative settings ({s['n_rates']} values) range from {s['min_rate_ppm']:.4g} to {s['max_rate_ppm']:.4g} "
                             f"µs s⁻¹; their signs {'differ' if s['signs_differ'] else 'agree'}; over the firing span the range amounts "
                             f"to {_against(s['effect_s'], tol)[0]} s.")
            if i['state'] == 'not_applicable':
                lines.append('- unconverged_iteration: not_applicable (the iteration converged).')
            else:
                parts = []
                if i.get('trial_difference_effect_s') is not None:
                    parts.append(f"Largest difference between the rate to be written and a trial rate: {i['max_trial_difference_ppm']:.4g} "
                                 f"µs s⁻¹, {_against(i['trial_difference_effect_s'], tol)[0]} s over the firing span.")
                if i.get('last_residual_effect_s') is not None:
                    parts.append(f"Residual rate at the last trial: {i['last_residual_ppm']:.4g} µs s⁻¹, "
                                 f"{_against(i['last_residual_effect_s'], tol)[0]} s over the firing span.")
                lines.append(f"- unconverged_iteration: {i['state']}{missing(i)}." + ''.join(' ' + part for part in parts))
        lines.extend(record['warnings'])
        if report.get('status') == 'supported' and record['status'] != 'not_blocked':
            lines.append('This result is supported. An export of it goes through the verified path, which this eligibility does not restrict.')
            lines.append('A verified log still requires that the read-back of the written file passes.')
        mapping = record.get('applied_mapping')
        if mapping:
            rate = f"{mapping['rate_ppm']} µs s⁻¹ ({mapping['rate_source']})"
            # report.md is written before any preview is drawn, and a run can leave the preview out.
            lines.append('Mapping in use: ' + {
                'corrected_laser_log': f'corrected_laser_log.csv applies {rate}.',
                'provisional_laser_log': f'provisional_laser_log.csv applies {rate}.',
                'display_only': f'no corrected or provisional log was written. A preview of this run, if one is drawn, displays {rate}.',
                'none': 'no corrected or provisional log was written, and a preview of this run, if one is drawn, displays no rate as applied.'}[mapping['kind']])
            rejected = mapping.get('rejected_candidate')
            if rejected:
                lines.append(f"Rejected candidate for inspection: {rejected['rate_ppm']} µs s⁻¹ ({rejected['rate_source']}). "
                             'It is not written to a corrected or provisional log and is not drawn as an applied correction.')
        lines.append(record['definitions'])
    except _TEXT_ERRORS:
        pass
    return lines


def _markdown(report, blocks=None):
    lines = ['# Timing diagnostic report', '', f"Status: **{report['status']}**",
             f"Version: {__version__}", f"Rate estimate (µs s⁻¹): {report.get('rate_ppm')}", *_rate_lines(report),
             f"Verified corrected log available: {report['correction_exported']}",
             f"Usable file: {report.get('usable_file') or 'none'}",
             f"Provisional (not verified) log: {report.get('provisional_file') or 'none'}", '', report.get('reason', '')]
    if report.get('notes'):
        lines.extend(['', '## Notes', ''] + [f'- {note}' for note in report['notes']])
    lines.extend(['', '## Input diagnostic checks', ''])
    for k, v in report.get('checks', {}).items():
        lines.append(f"- {k}: {'pass' if v else 'not met'}")
        if not v:
            lines.extend(f'  - {text}' for text in _check_detail(k, report, blocks))
    if not report.get('checks'):
        lines.extend(_stopped_before_checks(report))
    lines.extend(_distance_lines(report))
    screen = report.get('laser_log_screen') or {}
    if screen.get('evaluated'):
        lines.extend(['', '## Laser-log line-change screen', '',
                      f"Limit: {screen['limit_s']} s and {screen['scale_multiple']:g} times the scaled median absolute deviation "
                      f"of up to {screen['neighbours_each_side']} neighbouring intervals on each side.",
                      f"Largest deviation outside flagged intervals (s): {screen['largest_unflagged_abs_deviation_s']}"])
        if screen['limit_s'] != screen['default_limit_s']:
            lines.append(f"The limit was changed from the default {screen['default_limit_s']} s; "
                         f"{screen['n_flagged_at_default_limit']} interval(s) would be flagged at the default limit.")
        if screen['flagged']:
            lines.extend(f"- after firing interval {f['after_interval']}: {f['interval_s']:.3f} s, "
                         f"{f['deviation_s']:+.3f} s from the median of its neighbours" for f in screen['flagged'])
            lines.append(screen['interpretation'])
        else:
            lines.append('No line-change interval was flagged.')
    boundary = report.get('record_boundary_blocks') or {}
    if boundary.get('baseline') or boundary.get('closure') or boundary.get('in_sensitivity'):
        def listed(rows):
            return ', '.join(f"block {r['block']} ({r['side']} of record, {r['distance_to_record_boundary_s']:.3f} s from the boundary"
                             + (', excluded by the option' if r['excluded_from_fit_by_option'] else '') + ')' for r in rows) or 'none'
        def variants(mapping):
            return '; '.join(f"{k}: block(s) {', '.join(str(b) for b in v)}" for k, v in sorted(mapping.items())) or 'none'
        lines.extend(['', '## Blocks at a record boundary', '', boundary['definition'] + '.',
                      f"Policy: {boundary['policy']}", 'Initial fit: ' + listed(boundary['baseline']),
                      'Iterated fit: ' + listed(boundary['closure']),
                      'Stability alternatives: ' + variants(boundary.get('in_sensitivity') or {})])
        if boundary['policy'] == 'exclude':
            lines.extend(['Excluded from the reported fits by the declared option: '
                          + (', '.join(f'block {b}' for b in boundary['excluded_blocks']) or 'none'),
                          'Excluded in stability alternatives: ' + variants(boundary.get('excluded_in_sensitivity') or {}),
                          'An excluded block must still agree with the fitted line. When a block is excluded from a reported fit '
                          'or a stability alternative, an exported log remains provisional. '
                          + boundary['limitation']])
        else:
            lines.append('The search-edge check applies to these blocks. If a peak lies at a boundary imposed by the record, the '
                         'diagnosis is repeated without the scan lines within the guard of that end of the record (see record_edge_lines '
                         'in the report); if the check is still not met, exclude_record_boundary_blocks is the declared alternative.')
    selection = report.get('signal_selection')
    if selection:
        lines.extend(['', '## Matching-signal selection', '', selection['criterion'] + '.',
                      'Selected: ' + ' + '.join(selection['selected_columns'])])
        lines.extend(f"- {' + '.join(c['columns'])}: peak {c['peak']}" for c in selection['candidates'])
    if report.get('short_acquisition'):
        short = report['short_acquisition']
        lines.extend(['', '## Conditional short-acquisition analysis', '', short['explanation'],
                      f"Blocks executed: {short['executed_blocks']}; usable fitted blocks: {short['usable_blocks']}",
                      'Unavailable sensitivity configurations: ' + ', '.join(short['unavailable_sensitivity_configs'])])
        lines.extend(f"- {k}: {v['state']}" + ('; ' + v['reason'] if v.get('reason') else '')
                     for k,v in short['check_execution'].items())
    fallback = report.get('alignment_fallback') or {}
    if fallback.get('attempted'):
        lines.extend(['', '## Separate input-placement fallback', '', fallback['note'],
                      'Candidate failed checks: ' + ', '.join(k for k,v in fallback['diagnostic'].get('checks', {}).items() if not v),
                      'Strict diagnostic fields above are unchanged.'])
    post = report.get('post_export_verification')
    if post:
        lines.extend(['', '## Actual exported CSV verification', '',
                      f"Residual estimate (µs s⁻¹): {post['residual_rate_ppm']}",
                      f"Single-pass residual (µs s⁻¹): {post['residual_baseline_rate_ppm']}",
                      f"Coverage fraction: {post['coverage_fraction']}",
                      f"Largest held-out residual (s): {post['heldout_max_s']}",
                      'The full diagnostic read the written CSV. No new automatic constant alignment was used.', ''])
        for k, v in post['checks'].items():
            lines.append(f"- {k}: {'pass' if v else 'not met'}")
            if not v:
                lines.extend(f'  - {text}' for text in _readback_detail(k, post, report))
    lines.extend(_eligibility_lines(report))
    if report.get('provisional_file'):
        readback = report.get('provisional_readback') or {}
        qualified = report.get('status') == QUALIFIED_STATUS and not report.get('provisional_failed_checks')
        # A reference rate written after the read-back of the candidate failed is another file: the failed stage is that of the candidate.
        takeover = report.get('provisional_failure_stage') == 'readback' and report.get('provisional_rate_source') == 'user_reference_rate'
        lines.extend(['', '## Provisional corrected log (not verified)', '',
                      f"File: {report['provisional_file']}",
                      f"Rate applied (µs s⁻¹): {report.get('provisional_rate_ppm')} ({report.get('provisional_rate_source')})",
                      f"Mapping: {report.get('provisional_mode')}",
                      f"Failed stage: {report.get('provisional_failure_stage')}"
                      + (' of the candidate log written from the acquisition estimate (not this file)' if takeover else ''),
                      ('Failed checks: none. Provisional because the declared option exclude_record_boundary_blocks was used.'
                       if qualified else
                       ('Failed read-back checks of that candidate log: ' if takeover else 'Failed checks: ')
                       + (', '.join(report.get('provisional_failed_checks') or []) or str(report.get('reason') or 'see above'))),
                      f"Fixed-mapping read-back{' of this provisional log' if takeover else ''}: {readback.get('status')}; "
                      f"residual rate of the written file (µs s⁻¹): {readback.get('residual_rate_ppm')}",
                      (f"Read-back not run: {readback.get('reason')}" if readback.get('status') == 'not_run' else ''),
                      ('The written timestamps were checked against the stated mapping. All timing checks passed with the declared option, '
                       'which does not apply the search-edge criterion to an excluded block, so this file is a provisional correction '
                       'for the user to inspect, not a verified one.' if qualified else
                       'The written timestamps were checked against the stated mapping; the timing evidence did not pass, '
                       'so this file is a provisional correction for the user to inspect, not a verified one.'), ''])
    lines.extend(['', '## Reconstruction handoff', '', report.get('correction_note', 'No corrected file requested. ' + EXPORT_HINT),
                  '', 'Original signal intensities and input files are unchanged.',
                  'Passing checks describes internal timing evidence, not independent spatial accuracy.',
                  'This command has not tested these bytes in iolite/HDIP. Use iolite-export for ESL formatting and timing checks, then verify the exact file in the receiving application.',
                  'Do not use candidate files from failed or INCOMPLETE runs.', ''])
    return '\n'.join(lines)


def _write_provisional(out, ms, laser, report, options, analysis, reference_rate_ppm=None, eligibility=None,
                       readback_failed=None):
    """Write a provisional (not verified) corrected log when the input checks did not all pass.

    The diagnosis is unchanged. The log applies the rate estimated from this
    acquisition when that estimate is eligible for provisional export.
    Otherwise it applies a user-supplied reference rate, as a rate-only mapping
    that takes no fitted constant from the estimate, and without a reference
    rate no provisional log is written. The written timestamps are checked against
    the stated mapping, and the written file is re-diagnosed for information only.
    The returned manifest marks the file as provisional and lists the failed
    checks. ``readback_failed`` lists the failed read-back checks when the
    input diagnostic passed and the read-back of its candidate did not.
    """
    fallback = report.get('alignment_fallback') or {}
    fit_report = fallback['diagnostic'] if fallback.get('adopted') else report
    failed = [k for k, v in (fit_report.get('checks') or {}).items() if not v]
    rate, source = fit_report.get('rate_ppm'), 'input_position_fallback' if fallback.get('adopted') else 'acquisition_estimate'
    # A withheld estimate stays in the report with its failed checks; only the log is not written from it.
    withheld = None if rate is None or eligibility is None or eligibility['status'] == 'not_blocked' else rate
    if withheld is not None:
        rate = None
    if rate is None and reference_rate_ppm is not None:
        rate, source = float(reference_rate_ppm), 'user_reference_rate'
    if withheld is not None:
        eligibility['message'] = _withheld_message(eligibility, reference_used=rate is not None)
    if rate is None:
        note = eligibility['message'] if withheld is not None else (
            'No provisional log: no rate could be estimated from this acquisition and no reference rate '
            'was given (--reference-rate-ppm). Resolve the failed condition, declare the starting alignment, '
            'or supply a reference rate measured on the same instrument pair.')
        report['provisional_note'] = note
        report['correction_note'] = 'No verified corrected log because the timing checks did not pass. ' + note
        return None
    requested = options.get('mode', 'rate_only')
    plan_note = ''
    closure = fit_report.get('closure') or {}
    affine_available = bool(closure.get('converged') and closure.get('residual_fit'))
    if source in ('acquisition_estimate', 'input_position_fallback') and (requested == 'rate_only' or affine_available):
        # Configuration errors were rejected earlier; any error here is not a reason to fall back.
        plan = export_plan(ms, laser, fit_report, options)
    else:
        if requested == 'affine':
            plan_note = (' An affine mapping would need a fitted constant of the estimate that was not eligible, and a reference '
                         'rate takes none; a rate-only provisional log was written instead.' if withheld is not None else
                         ' An affine mapping needs the fitted constant of a converged iteration, which is not available here; '
                         'a rate-only provisional log was written instead.')
        plan = dict(mode='rate_only', offset_seconds='0', coordinate_reference=None,
                    initial_alignment_offset_s=(fit_report.get('alignment') or {}).get('offset_s'))
    candidate = out/'provisional_laser_log.csv'
    receipt = export_corrected(laser, candidate, rate, offset_seconds=plan['offset_seconds'])
    _check_written_mapping(receipt, rate, plan, candidate)
    readback = dict(status='not_run',
                    reason='No starting alignment is available for a fixed-mapping read-back; no new alignment is guessed.')
    if (fit_report.get('alignment') or {}).get('offset_s') is not None:
        # Byte or mapping errors raised here are execution errors, not informational read-back results.
        post, rerun, reblocks = verify_written_log(ms, laser, fit_report, candidate, receipt, plan, analysis)
        write_json(out/'provisional_readback_report.json', rerun)
        write_blocks(out, reblocks, 'provisional_readback_blocks')
        readback = dict(status='passed' if post['passed'] else 'timing_checks_failed',
                        residual_rate_ppm=post['residual_rate_ppm'],
                        residual_baseline_rate_ppm=post['residual_baseline_rate_ppm'],
                        checks=post['checks'], failed_checks=post['failed_checks'],
                        interpretation='Re-diagnosis of the written provisional file with the mapping held fixed')
    _check_written_mapping(receipt, rate, plan, candidate)
    what = ('the user-supplied reference rate' if source == 'user_reference_rate' else
            'the rate estimated at the original input placement' if fallback.get('adopted') else
            'the rate estimated from this acquisition')
    if readback_failed is not None:
        failed = list(readback_failed)
        note = ('Provisional, NOT verified: the input diagnostic passed, but the read-back of the candidate written from the '
                'acquisition estimate did not (' + ', '.join(failed) + '). '
                f'This log applies {what} ({rate:.3f} µs s⁻¹) as a {plan["mode"]} mapping. '
                'Inspect the reconstruction before use and state the failed checks with any result.' + plan_note)
    elif fit_report.get('status') == QUALIFIED_STATUS and not failed:
        note = ('Provisional, NOT verified: all timing checks passed, but only with the declared option exclude_record_boundary_blocks. '
                + str(fit_report.get('reason') or '') + ' '
                f'This log applies {what} ({rate:.3f} µs s⁻¹) as a {plan["mode"]} mapping. '
                'Inspect the reconstruction before use and state the option with any result.' + plan_note)
    else:
        note = ('Provisional, NOT verified: the input timing checks did not all pass (' +
                (', '.join(failed) or str(report.get('reason') or 'see report')) + '). '
                f'This log applies {what} ({rate:.3f} µs s⁻¹) as a {plan["mode"]} mapping. '
                'Inspect the reconstruction before use and state the failed checks with any result.' + plan_note)
    if plan['mode'] == 'rate_only':
        note += ' Determine the constant offset in the reconstruction importer; do not apply the rate a second time.'
    if withheld is not None:
        shown = f'{withheld:.3f}' if isinstance(withheld, (int, float)) else str(withheld)
        origin = 'at the original input placement' if fallback.get('adopted') else 'from this acquisition'
        note += f' The rate estimated {origin} ({shown} µs s⁻¹) was not written to a provisional log: ' + '; '.join(eligibility['reasons']) + '.'
    if fallback.get('attempted'):
        note += ' ' + fallback['note']
    if eligibility is not None and eligibility.get('warnings'):
        note += WARNINGS_POINTER
    failure_stage = ('readback' if readback_failed is not None else
                     'alignment_fallback' if fallback.get('adopted') else 'input_diagnostic')
    report.update(provisional_file=str(candidate), provisional_rate_ppm=rate, provisional_rate_source=source,
                  provisional_mode=plan['mode'], provisional_export_plan=plan,
                  provisional_export_verification=receipt, provisional_readback=readback,
                  provisional_failed_checks=failed, provisional_failure_stage=failure_stage,
                  provisional_note=note, correction_note=note)
    return dict(version=__version__, provisional=True, verified=False, mode=plan['mode'],
                rate_ppm=rate, rate_source=source, failed_checks=failed, failure_stage=failure_stage,
                alignment_fallback=fallback, input_diagnostic_status=report.get('input_diagnostic_status'),
                diagnostic_status=report['status'], reason=report.get('reason'), readback_status=readback['status'],
                output_file=str(candidate), output_sha256=receipt['output_sha256'], export_plan=plan,
                full_export_diagnostic_passed=False, importer_roundtrip_tested=False,
                laser_log_screen=_screen_summary(report), instruction=note)


def _publish_provisional_from_candidate(out, report, candidate, receipt, plan, post, eligibility=None):
    """After a failed read-back, publish the already checked candidate bytes as provisional.

    The correction is not applied again. A candidate whose written rate or
    constant differs from the plan is not published (the run keeps exit code
    2). Changed bytes are an execution error and are not downgraded: they
    raise, so the run ends with exit code 1 and INCOMPLETE.json. The byte
    checks come first. The eligibility is that of the estimate the candidate
    was written from; the residual rate of the read-back is not a new
    correction. A candidate that is not eligible stays for diagnosis only.
    """
    _check_written_bytes(receipt, candidate)
    if digest(candidate) != post['diagnostic_input_sha256']:
        raise InputError('Candidate changed after the read-back diagnostic')
    if eligibility is not None and eligibility['status'] != 'not_blocked':
        eligibility['message'] = _withheld_message(eligibility)
        report['provisional_note'] = eligibility['message']
        return None
    try:
        _check_written_plan(receipt, report['rate_ppm'], plan)
    except InputError as exc:
        report['provisional_note'] = f'No provisional log: {exc}. The candidate is retained only for diagnosis.'
        return None
    payload = candidate.read_bytes()
    final = out/'provisional_laser_log.csv'
    with final.open('xb') as f:
        f.write(payload)
    if digest(final) != receipt['output_sha256']:
        raise InputError('Provisional file differs from the checked candidate bytes')
    failed = list(post['failed_checks'])
    note = ('Provisional, NOT verified: the input diagnostic passed, but the read-back of the written file did not ('
            + ', '.join(failed) + '). '
            f"This log keeps the checked candidate bytes ({float(report['rate_ppm']):.3f} µs s⁻¹, {plan['mode']} mapping); "
            'the correction is not applied again. Inspect the reconstruction before use and state the failed checks with any result.')
    if plan['mode'] == 'rate_only':
        note += ' Determine the constant offset in the reconstruction importer; do not apply the rate a second time.'
    if eligibility is not None and eligibility.get('warnings'):
        note += WARNINGS_POINTER
    report.update(provisional_file=str(final), provisional_rate_ppm=report['rate_ppm'],
                  provisional_rate_source='acquisition_estimate', provisional_mode=plan['mode'],
                  provisional_export_plan=plan, provisional_export_verification=dict(receipt, output_path=str(final)),
                  provisional_readback=dict(status='timing_checks_failed', residual_rate_ppm=post['residual_rate_ppm'],
                                            residual_baseline_rate_ppm=post['residual_baseline_rate_ppm'],
                                            checks=post['checks'], failed_checks=failed),
                  provisional_failed_checks=failed, provisional_failure_stage='readback',
                  provisional_note=note, correction_note=note)
    return dict(version=__version__, provisional=True, verified=False, mode=plan['mode'],
                rate_ppm=report['rate_ppm'], rate_source='acquisition_estimate', failed_checks=failed,
                failure_stage='readback', diagnostic_status=report['status'],
                input_diagnostic_status=report.get('input_diagnostic_status'), reason=report.get('reason'),
                readback_status='timing_checks_failed', output_file=str(final), output_sha256=receipt['output_sha256'],
                export_plan=plan, full_export_diagnostic_passed=False, importer_roundtrip_tested=False,
                laser_log_screen=_screen_summary(report), instruction=note)


def _preview_after_run(out, channel, sample_label=True):
    """Default picture drawn after report.json; a drawing failure is explicit."""
    try:
        from .preview import render_preview_pair
        summary = render_preview_pair(out, channel) if sample_label else render_preview_pair(out, channel, sample_label=False)
        return dict(status='written', png=summary['png'], json=summary['json'],
                    channel=summary['channel'], rate_fitted=summary['rate_fitted'])
    except Exception as exc:  # the diagnosis is complete; the preview must not undo it
        failure = dict(status='failed', error=f'{type(exc).__name__}: {exc}')
        if isinstance(exc,IntegrityError):
            write_json(out/'INCOMPLETE.json',dict(status='tool_or_input_error',error=str(exc),usable_file=None,
                       note='Byte integrity failed during preview; no output from this run is accepted'))
        else:
            try:
                from .preview import render_raw_diagnostic
                config=json.loads((out/'config.json').read_text(encoding='utf-8'))
                report=json.loads((out/'report.json').read_text(encoding='utf-8'))
                name='preview_raw' if (out/'preview_pair.png').exists() or (out/'preview_pair.json').exists() else 'preview_pair'
                raw=render_raw_diagnostic(out,config,report['input_sha256'],'Requested preview failed: '+str(exc),name=name)
                failure.update(status='fallback_written',png=raw['png'],json=raw['json'],channel=raw['channel'],rate_fitted=False)
            except Exception as fallback_exc:
                failure['fallback_error']=str(fallback_exc)
                if isinstance(fallback_exc,IntegrityError) and not (out/'INCOMPLETE.json').exists():
                    write_json(out/'INCOMPLETE.json',dict(status='tool_or_input_error',error=str(fallback_exc),usable_file=None))
        write_json(out/'preview_error.json', failure)
        print(json.dumps(dict(preview=failure), ensure_ascii=False), file=sys.stderr)
        return failure


def _raw_error_preview(out, config, before, exc, partial_signal=False):
    """Keep the execution error and inspect original bytes only, never candidate outputs."""
    try:
        from .preview import render_raw_diagnostic, render_partial_signal
        if any(digest(config[k]['path']) != v for k,v in before.items()):
            return
        out.mkdir(parents=True, exist_ok=True)
        if not (out/'config.json').exists():
            write_json(out/'config.json', config)
        if not (out/'INCOMPLETE.json').exists():
            write_json(out/'INCOMPLETE.json', dict(status='tool_or_input_error', error=str(exc), usable_file=None))
        if partial_signal and (config['ms'].get('format') != 'csv' or config['ms'].get('time_kind') != 'numeric'):
            return    # partial inspection exists only for an explicit numeric CSV mapping; the input error is reported by the caller
        render=render_partial_signal if partial_signal else render_raw_diagnostic
        render(out, config, before, 'Execution error: '+str(exc))
    except Exception as preview_exc:
        print(json.dumps(dict(preview=dict(status='failed', error=str(preview_exc))),ensure_ascii=False),file=sys.stderr)


def _run_segments(config, out, before, segmentation, export, preview, options, alignment_fallback, sample_label=True,
                  tolerance_s=None, tolerance_um=None):
    from .preview import render_raw_diagnostic
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'config.json',config)
    entries=[]
    for item in segmentation['segments']:
        dest=out/f"segment_{item['segment']:03d}"
        child=json.loads(json.dumps(config))
        child['laser']['record_range']=[item['start_record'], item['end_record']]
        if item['sample']:
            child['laser']['sample_selector']=item['sample']
        if 'alignment_offset_s' in child['analysis']:
            child['analysis']['alignment_offset_s'] += item['origin_shift_s']
        entry=dict(item,output=str(dest),declared_alignment_offset_s=child['analysis'].get('alignment_offset_s'))
        if item.get('on_record_count') == 0 and child['laser'].get('format','laser_state_csv') == 'laser_state_csv':
            dest.mkdir()
            note='No firing On records in this fragment; metadata retained, no sample image or corrected log is invented.'
            fragment=dict(status='no_firing_fragment',reason=note,source=item)
            write_json(dest/'report.json',fragment)
            entry.update(exit_code=2,status='no_firing_fragment',reason=note,preview=None)
        else:
            try:
                # Each fragment resolves its tolerance and its provisional export eligibility on its own firing intervals.
                code, report=run(child,dest,export,preview=preview,alignment_fallback=alignment_fallback,
                                 preview_sample_label=sample_label,_fallback_input_offset_s=item['origin_shift_s'],
                                 tolerance_s=tolerance_s,tolerance_um=tolerance_um)
                resolution=report.get('tolerance_resolution') or {}
                # The tolerance in seconds of this fragment, and the speed that converted a request in micrometres (None for seconds).
                entry.update(exit_code=code,status=report['status'],rate_ppm=report.get('rate_ppm'),
                             preview=report.get('preview'),provisional_file=report.get('provisional_file'),usable_file=report.get('usable_file'),
                             alignment_fallback=report.get('alignment_fallback'),
                             provisional_export_eligibility=(report.get('provisional_export_eligibility') or {}).get('status'),
                             resolved_tolerance_s=resolution.get('resolved_tolerance_s'),
                             tolerance_conversion_speed_um_s=resolution.get('speed_um_s'))
            except (InputError,ValueError,KeyError,OSError) as exc:
                entry.update(exit_code=1,status='input_or_tool_error',error=str(exc))
        entries.append(entry)
    if any(digest(config[k]['path']) != v for k,v in before.items()):
        write_json(out/'INCOMPLETE.json',dict(status='tool_or_input_error',error='Input changed during segmented run'))
        raise InputError('Input changed during segmented run')
    code=1 if any(v['exit_code']==1 for v in entries) else 2
    report=dict(tool='la-lagcheck',version=__version__,status='segmented_with_errors' if code==1 else 'segmented_not_jointly_verified',
                reason='Contiguous sample/sequence fragments were processed separately; no joint correction was made.',
                rate_ppm=None,correction_exported=False,usable_file=None,provisional_file=None,
                exported_log_diagnostic_rerun=False,input_sha256=before,segments=entries)
    original_check=dict(status='requires_processing',reasons=['sample_or_sequence_segmentation'],
                        assessment='Source-record inspection; fragments are not one joint acquisition')
    report['original_input_check']=original_check
    segmentation['original_input_check']=original_check
    write_json(out/'segments.json',dict(segmentation,segments=entries))
    write_json(out/'input_processing.json',dict(segmentation,segments=entries))
    write_json(out/'report.json',report)
    (out/'report.md').write_text('# Segmented input\n\n'+report['reason']+'\n\n'+
        '\n'.join(f"- Segment {v['segment']}, records {v['start_record']} to {v['end_record']}: {v['status']}" for v in entries),encoding='utf-8')
    if preview is not False:
        try:
            fragment_summary='; '.join(f"segment {v['segment']}: {v['status']}" for v in entries)
            summary=render_raw_diagnostic(out,config,before,report['reason']+' '+fragment_summary+'. See segments.json and per-fragment previews.',details=entries)
            report['preview']=dict(status='written',png=summary['png'],json=summary['json'],channel=summary['channel'],rate_fitted=False)
        except Exception as exc:
            report['preview']=dict(status='failed',error=str(exc)); code=1
            write_json(out/'preview_error.json',report['preview'])
    return code,report


def screen_log(config_path, limit_s=1.0):
    """Screen the laser log named in a run configuration; no signal file is read.

    Contiguous sample/sequence fragments are screened separately, as in a run.
    Exit code 0 means no flagged interval and 2 at least one.
    """
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    if not isinstance(config, dict) or not isinstance(config.get('laser'), dict) or not isinstance(config['laser'].get('path'), str):
        raise InputError('Configuration must be a JSON object with laser.path')
    if isinstance(limit_s, bool) or not isinstance(limit_s, (int, float)) or not math.isfinite(limit_s) or limit_s <= 0:
        raise InputError('The limit must be a positive number of seconds')
    path = (config_path.parent/config['laser']['path']).resolve()
    if not path.is_file():
        raise InputError('laser source does not exist')
    laser_config = {k: v for k, v in config['laser'].items() if k != 'path'}
    laser_config.setdefault('zero_duration_policy', 'exclude')
    from .processing import inspect_segments
    from .estimator import screen_line_changes
    before = digest(path)
    segmentation = inspect_segments(path, laser_config)
    several = len(segmentation['segments']) > 1
    results = []
    for item in segmentation['segments']:
        child = dict(laser_config)
        entry = dict(segment=item['segment'], records=[item['start_record'], item['end_record']])
        if several:
            child['record_range'] = [item['start_record'], item['end_record']]
            if item['sample']:
                child['sample_selector'] = item['sample']
        if item.get('on_record_count') == 0 and child.get('format', 'laser_state_csv') == 'laser_state_csv':
            results.append(dict(entry, status='no_firing_fragment'))
            continue
        laser = load_laser(path, child)
        results.append(dict(entry, n_firing_intervals=len(laser.lines),
                            screen=screen_line_changes(laser.lines, float(limit_s))))
    if digest(path) != before:
        raise InputError('Laser log changed during screening')
    flagged = sum(len(r['screen']['flagged']) for r in results if 'screen' in r)
    evaluated = sum(1 for r in results if 'screen' in r and r['screen']['evaluated'])
    unevaluated = len(results) - evaluated
    # Exit code 2 also when nothing could be screened, so that an empty result is not read as a clean log.
    status = 'flagged' if flagged else ('none_flagged' if evaluated else 'not_evaluated')
    return (0 if status == 'none_flagged' else 2), dict(tool='la-lagcheck', version=__version__, status=status, laser=str(path),
                                                        laser_sha256=before, flagged_intervals=flagged,
                                                        evaluated_segments=evaluated, unevaluated_segments=unevaluated,
                                                        segments=results)


def run(config_path, output, export=False, export_mode=None, ms_first_sample_time=None, reference_note=None, preview=None,
        provisional=None, reference_rate_ppm=None, alignment_fallback=True, preview_sample_label=True, _fallback_input_offset_s=0.,
        tolerance_s=None, tolerance_um=None):
    if isinstance(config_path, dict):
        config = json.loads(json.dumps(config_path))
        base = Path.cwd()
    else:
        config_path = Path(config_path).resolve()
        config = json.loads(config_path.read_text(encoding='utf-8-sig'))
        base = config_path.parent
    # tolerance_resolution is the record that an earlier run wrote into its config.json; it is not a setting.
    if set(config)-{'ms','laser','analysis','export','tolerance_resolution'} or any(k not in config for k in ('ms','laser','analysis')):
        raise InputError('Configuration requires ms, laser and analysis, with optional export settings')
    options = dict(config.get('export', {}))
    for key, value in [('mode', export_mode), ('ms_first_sample_at_laser_time', ms_first_sample_time), ('reference_note', reference_note),
                       ('provisional', provisional), ('reference_rate_ppm', reference_rate_ppm)]:
        if value is not None:
            options[key] = value
    config['export'] = options
    config['laser'].setdefault('zero_duration_policy','exclude')
    # Provisional output (new in 0.2.0) is on by default with --export; it never
    # changes the diagnosis or the verified export below.
    provisional = options.get('provisional', True)
    reference_rate_ppm = options.get('reference_rate_ppm')
    if not isinstance(provisional, bool):
        raise InputError('export.provisional must be true or false')
    _validate_reference_rate(reference_rate_ppm)
    plan_options = {k: v for k, v in options.items() if k not in ('provisional', 'reference_rate_ppm')}
    # Conflicting or invalid tolerance requests stop here, before any file is read. A tolerance in micrometres
    # is converted to seconds further down, once the firing intervals of the selected acquisition are known.
    request = read_request(config['analysis'], tolerance_s, tolerance_um, config.pop('tolerance_resolution', None))
    paths = {}
    for kind in ('ms','laser'):
        if not isinstance(config[kind], dict) or not isinstance(config[kind].get('path'), str):
            raise InputError(kind+'.path must be a filename')
        paths[kind] = (base/config[kind]['path']).resolve()
        if not paths[kind].is_file():
            if kind=='laser' and preview is not False and paths.get('ms') and paths['ms'].is_file():
                raw_out=Path(output).resolve()
                if not raw_out.exists():
                    _raw_error_preview(raw_out,config,{'ms':digest(paths['ms'])},InputError('laser source does not exist'))
            raise InputError(kind+' source does not exist')
        config[kind]['path'] = str(paths[kind])
    out = Path(output).resolve()
    if out.exists():
        raise InputError('Output directory must not already exist; originals/results are never overwritten')
    if export:
        _reject_known_corrected_input(paths['laser'])
    before = {k: digest(v) for k,v in paths.items()}
    ms_config = {k:v for k,v in config['ms'].items() if k not in ('path','signal_selection')}
    # Default: compare the sum and each configured column by starting-alignment contrast (CSV formats).
    selection_mode = config['ms'].get('signal_selection', 'sum' if ms_config.get('format')=='netcdf' else 'highest_alignment_peak')
    if selection_mode not in SELECTION_MODES:
        raise InputError("ms.signal_selection must be 'sum' or 'highest_alignment_peak'")
    if selection_mode != 'sum' and ms_config.get('format')=='netcdf':
        raise InputError('ms.signal_selection is available for the CSV signal formats only')
    try:
        if ms_config.get('format')=='netcdf':
            from .netcdf import load_netcdf_signal
            ms = load_netcdf_signal(paths['ms'], ms_config)
        else:
            ms = load_signal(paths['ms'], ms_config)
    except (InputError,ValueError,OSError) as exc:
        if preview is not False:
            _raw_error_preview(out,config,before,exc,partial_signal=True)
        raise
    try:
        from .processing import inspect_segments
        segmentation=inspect_segments(paths['laser'],{k:v for k,v in config['laser'].items() if k!='path'})
        if segmentation['source_sha256'] != before['laser']:
            raise IntegrityError('Laser bytes read for segmentation differ from the initial input snapshot')
        if len(segmentation['segments']) > 1:
            return _run_segments(config,out,before,segmentation,export,preview,options,alignment_fallback,preview_sample_label,
                                 tolerance_s,tolerance_um)
        laser = load_laser(paths['laser'], {k:v for k,v in config['laser'].items() if k!='path'})
    except (InputError,ValueError,KeyError,OSError) as exc:
        if preview is not False:
            _raw_error_preview(out,config,before,exc)
        raise
    config['laser'].update(laser.metadata['config'])
    if export:
        _validate_export_options(options, laser)
    try:
        # The tolerance is resolved here, once, on the original log. analysis holds seconds only and is what the
        # diagnosis, the retry and every read-back receive; the settings used for reporting only do not reach them.
        analysis, tolerance_record, speed = resolve_tolerance(request, config['analysis'], laser)
        spot = spot_width(laser, config['analysis'])
        if request['unit'] == 'um' or request['overridden'] or request['saved']:
            # The saved config then holds the seconds that were used, so a rerun and iolite-export do not convert again.
            config['analysis'] = dict({k: v for k, v in config['analysis'].items() if k != 'tolerance_um'},
                                      tolerance_s=analysis['tolerance_s'])
            config['tolerance_resolution'] = tolerance_record
        selection = None
        if selection_mode == 'highest_alignment_peak':
            # Declared option: the saved config then names the columns actually
            # used, so previews and read-back reload the same matching signal.
            ms, selection = select_signal(ms, laser, analysis,
                                          lambda columns: load_signal(paths['ms'], dict(ms_config, signal_columns=columns)))
            config['ms']['signal_columns'] = list(selection['selected_columns'])
        # The saved config fixes the matching signal that was actually used: the columns
        # named there are summed as given, so rerunning the saved config does not select again.
        config['ms']['signal_selection'] = 'sum'
        report, blocks = diagnose(ms, laser, analysis)
        if selection is not None:
            report['signal_selection'] = selection
        fallback, fallback_blocks = try_alignment_fallback(ms, laser, report, analysis, diagnose,
            enabled=alignment_fallback, provisional=provisional, input_offset_s=_fallback_input_offset_s)
    except (InputError,ValueError,KeyError) as exc:
        if preview is not False:
            _raw_error_preview(out,config,before,exc)
        raise
    processed_status=report['status']
    processing=laser.metadata['input_processing']
    required=[]
    if processing['excluded_count']:
        required.append('zero_duration_intervals')
    if config['laser'].get('record_range') is not None:
        required.append('sample_or_sequence_segmentation')
    original_check=dict(status='requires_processing' if required else 'accepted',reasons=required,
                        assessment='Source-record inspection; not a second run of the timing diagnostic')
    processing.update(original_input_check=original_check,processed_input_diagnostic_status=processed_status,
                      publication_policy='processed inputs are provisional only' if required else 'normal two-stage verification')
    if required and processed_status=='supported':
        report['status']='processed_input_diagnostic_supported'
        report['reason']='The processed intervals support the model; the original input required '+', '.join(required)
    report.update(original_input_check=original_check,processed_input_diagnostic_status=processed_status)
    report['alignment_fallback'] = fallback
    report.update(tool='la-lagcheck', version=__version__, input_sha256=before,
                  input_metadata={'ms':ms.metadata, 'laser':laser.metadata},
                  input_diagnostic_status=report['status'], correction_exported=False,
                  importer_roundtrip_tested=False, exported_log_diagnostic_rerun=False,
                  receiver_interval_trajectory_agreement_verified=False,
                  usable_file=None, candidate_file=None)
    # Eligibility for provisional export, assessed for the diagnostic an export would use. It reads the diagnosis
    # and changes none of the fields above.
    fit_report, candidate_source = _selected_diagnostic(report)
    duration = float(laser.lines[-1]['End Time']-laser.lines[0]['Start Time'])
    eligibility = assess_eligibility(fit_report, duration, analysis['tolerance_s'], candidate_source)
    eligibility.update(tolerance_request=dict(unit=tolerance_record['requested_unit'], value=tolerance_record['requested_value']),
                       message=None)
    report.update(provisional_export_eligibility=eligibility, tolerance_resolution=tolerance_record)
    if any(digest(paths[k])!=v for k,v in before.items()):
        raise InputError('Input changed during analysis; no correction will be exported')
    out.mkdir(parents=True, exist_ok=False)
    provisional_manifest = None
    try:
        write_json(out/'config.json', config)
        write_json(out/'input_processing.json', laser.metadata['input_processing'])
        write_blocks(out, blocks)
        if fallback['attempted']:
            write_json(out/'alignment_fallback_report.json', fallback['diagnostic'])
            write_blocks(out, fallback_blocks, 'alignment_fallback_blocks')
        if export and report['status']=='supported':
            plan = export_plan(ms, laser, report, plan_options)
            report['export_plan'] = plan
            candidate = out/'candidate_laser_log.csv'
            receipt = export_corrected(laser, candidate, report['rate_ppm'], offset_seconds=plan['offset_seconds'])
            report['candidate_file'] = str(candidate)
            report['export_verification'] = receipt
            post, rerun, reblocks = verify_written_log(ms, laser, report, candidate, receipt, plan, analysis)
            report['exported_log_diagnostic_rerun'] = True
            report['post_export_verification'] = post
            write_json(out/'post_export_report.json', rerun)
            write_blocks(out, reblocks, 'post_export_blocks')
            if post['passed']:
                # Publish exactly the fully diagnosed on-disk bytes. Retain the
                # candidate for the audit; no file is moved or overwritten.
                payload = candidate.read_bytes()
                if hashlib.sha256(payload).hexdigest()!=post['diagnostic_input_sha256']:
                    raise InputError('Candidate changed before publication')
                final = out/'corrected_laser_log.csv'
                with final.open('xb') as f:
                    f.write(payload)
                final_parsed = load_laser(final, receipt['output_config'])
                if digest(final)!=post['diagnostic_input_sha256'] or final_parsed.metadata['input_sha256']!=post['diagnostic_input_sha256']:
                    raise InputError('Published CSV differs from the fully diagnosed bytes')
                receipt.update(output_path=str(final), output_sha256=digest(final))
                post['published_identical_bytes'] = True
                report['correction_exported'] = True
                report['usable_file'] = str(final)
                report['correction_note'] = (plan.get('import_instruction') if plan['mode']=='affine' else
                    'Rate-only corrected log: determine its constant offset in the reconstruction importer. '
                    'Before image use, verify that firing-interval endpoints and continuous-trajectory transition times '
                    'use the same corrected time coordinate after the offset. '
                    'Do not apply the original rate a second time. Importer compatibility remains untested.')
            else:
                report['status'] = 'export_qc_failed'
                report['reason'] = 'Actual exported CSV did not pass all residual timing checks'
                report['correction_note'] = 'No usable corrected file. The candidate CSV is retained only for diagnosis; do not import it as an accepted correction.'
                if provisional:
                    provisional_manifest = _publish_provisional_from_candidate(out, report, candidate, receipt, plan, post, eligibility)
                    if provisional_manifest is None and eligibility['message'] and reference_rate_ppm is not None:
                        # The candidate is not eligible; the reference rate is written as its own rate-only log.
                        provisional_manifest = _write_provisional(out, ms, laser, report, plan_options, analysis,
                                                                  reference_rate_ppm, eligibility, post['failed_checks'])
        elif export and provisional:
            provisional_manifest = _write_provisional(out, ms, laser, report, plan_options, analysis, reference_rate_ppm, eligibility)
        elif export:
            report['correction_note'] = 'No corrected log exported because timing checks did not pass'
        report['input_unchanged'] = all(digest(paths[k])==v for k,v in before.items())
        if not report['input_unchanged']:
            raise InputError('Input changed before completion')
        # One record of the mapping in use, read by the log manifest and by every preview of this run.
        eligibility['applied_mapping'] = _applied_mapping(report, eligibility, reference_rate_ppm)
        try:
            report['distance_scale'] = distance_scale(fit_report, fallback_blocks if fallback['adopted'] else blocks,
                                                      analysis['tolerance_s'], duration, speed, spot, eligibility,
                                                      'input_position_fallback' if fallback['adopted'] else 'main')
        except _TEXT_ERRORS:        # distances are reporting only and never undo a completed diagnosis
            report['distance_scale'] = None
        _annotate(report, reference_rate_ppm, export, provisional)
        write_json(out/'report.json', report)
        (out/'report.md').write_text(_markdown(report, blocks), encoding='utf-8')
        if provisional_manifest is not None:
            # Written last, like the verified manifest; it marks the file as provisional.
            write_json(out/'provisional_manifest.json', dict(provisional_manifest, input_sha256=before,
                       acquisition_identity=laser.metadata['acquisition_identity'],
                       provisional_export_eligibility=eligibility, tolerance_resolution=tolerance_record,
                       warnings=[text for text in (eligibility['message'] or eligibility['explanation'], *eligibility['warnings']) if text]))
        if report['correction_exported']:
            # The manifest is the last publication marker, after input hashes
            # and both human/machine-readable final reports are complete.
            write_json(out/'correction_manifest.json', dict(version=__version__, mode=plan['mode'],
                       input_sha256=before, output_sha256=digest(final), output_file=str(final),
                       export_plan=plan, full_export_diagnostic_passed=True,
                       acquisition_identity=laser.metadata['acquisition_identity'],
                       receiver_interval_trajectory_agreement_verified=False,
                       laser_log_screen=_screen_summary(report),
                       importer_roundtrip_tested=False, instruction=report['correction_note'],
                       **({'tolerance_resolution': tolerance_record} if 'tolerance_resolution' in config else {})))
        code = 0 if report['status']=='supported' else 2
    except Exception as exc:
        write_json(out/'INCOMPLETE.json', dict(status='tool_or_input_error', error=str(exc),
                   usable_file=None, note='Do not use any outputs from this incomplete run'))
        if preview is not False:
            _raw_error_preview(out,config,before,exc)
        raise
    if preview is not False:
        # Drawn only after report.json and any manifest exist. It reads the run
        # directory back; the written report, blocks and logs are not touched.
        report['preview'] = _preview_after_run(out, None if preview in (True,None) else preview, preview_sample_label)
        if report['preview']['status'] != 'written':
            code=1
        elif eligibility['message'] and eligibility['applied_mapping']['kind'] == 'none':
            # The uncorrected view exists now, so the returned notes name it. report.json was written before it was drawn.
            drawn = _withheld_message(eligibility, view_available=True)
            report['notes'] = [drawn if note == eligibility['message'] else note for note in report['notes']]
    return code, report


def main(argv=None):
    _configure_utf8_stdio()
    parser = argparse.ArgumentParser(description='LA-LagCheck: per-acquisition laser/MS lag rate, checks and verified corrected log')
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run', help='Diagnose and optionally correct one acquisition from a JSON configuration')
    p.add_argument('config'); p.add_argument('--output', required=True)
    p.add_argument('--export', action='store_true', help='Export and fully re-diagnose the written CSV; publish only after checks pass')
    p.add_argument('--export-mode', choices=('rate_only','affine'))
    p.add_argument('--ms-first-sample-time', help='Explicit time of first stored MS sample in the laser output timestamp coordinate (affine only)')
    p.add_argument('--reference-note', help='Where that first-sample time reference was verified')
    p.add_argument('--no-provisional', dest='provisional', action='store_false', default=None,
                   help='Disable alignment fallback and provisional (not verified) log export')
    p.add_argument('--no-alignment-fallback', dest='alignment_fallback', action='store_false', default=True,
                   help='Do not estimate a separate provisional rate at the input placement after eligible automatic-alignment failures')
    p.add_argument('--reference-rate-ppm', type=float, default=None,
                   help='Rate in µs s⁻¹ to use for a provisional log when no rate can be estimated from this acquisition, '
                        'or when the estimate is not eligible for provisional export '
                        '(for example the median rate measured on the same instrument pair)')
    p.add_argument('--tolerance-s', type=float, default=None,
                   help='Tolerance in seconds for this run. It replaces the tolerance of the configuration, whatever its unit')
    p.add_argument('--tolerance-um', type=float, default=None,
                   help='Tolerance in micrometres along the scan for this run. It is converted to seconds once, before the '
                        'diagnosis, with the largest scan speed of the firing intervals, and replaces the tolerance of the '
                        'configuration. Give only one of --tolerance-s and --tolerance-um')
    p.add_argument('--preview', nargs='?', const=True, default=None, metavar='CHANNEL',
                   help='Select the channel for the default preview_pair.png. Actual published mappings take precedence; '
                        'otherwise show the current estimated or explicitly supplied rate, or raw alignment when no rate exists. '
                        'The first stored channel is the default. Failed checks remain visible; --no-preview disables drawing.')
    p.add_argument('--no-preview', dest='preview', action='store_const', const=False,
                   help='Explicitly disable the default visual preview')
    label_help = 'Leave the sample label of the laser log (or, without one, the laser file name) out of the title of preview_pair.png'
    p.add_argument('--no-sample-label', dest='sample_label', action='store_false', default=True, help=label_help)
    p = sub.add_parser('wizard', help='Guided MassHunter/laser-file selection without writing JSON')
    p.add_argument('--no-alignment-fallback',dest='alignment_fallback',action='store_false',default=True)
    p.add_argument('--no-provisional',dest='provisional',action='store_false',default=None)
    p.add_argument('--no-preview',dest='preview',action='store_false',default=True,
                   help='Disable the default visual preview')
    p.add_argument('--no-sample-label', dest='sample_label', action='store_false', default=True, help=label_help)
    p = sub.add_parser('preview', help='Draw the preview pair for a completed run directory without changing its results')
    p.add_argument('run_dir', help='Directory holding config.json and report.json of a completed run')
    p.add_argument('--channel', help='Signal column or variable to draw; default: the first stored channel')
    p.add_argument('--name', default='preview_pair', help='File stem inside the run directory (default preview_pair); existing files are never overwritten')
    p.add_argument('--no-sample-label', dest='sample_label', action='store_false', default=True, help=label_help)
    p = sub.add_parser('iolite-export', help='Convert a completed canonical run, verified or provisional, to ESL millisecond CSV, '
                                             'then recheck saved bytes under its fixed timing mapping')
    p.add_argument('canonical_run', help='Directory containing config.json, report.json and either correction_manifest.json with '
                                         'corrected_laser_log.csv or, for a provisional run, provisional_manifest.json with '
                                         'provisional_laser_log.csv; a provisional log stays marked as provisional')
    p.add_argument('--output', required=True, help='New result directory; existing directories are rejected')
    p = sub.add_parser('inspect-netcdf', help='List classic NetCDF metadata without guessing a mapping')
    p.add_argument('path')
    p = sub.add_parser('screen-log', help='Screen the line-change intervals of the laser log named in a run configuration; no signal file is read')
    p.add_argument('config')
    p.add_argument('--limit', type=float, default=1.0, help='Limit in seconds (default 1.0)')
    args = parser.parse_args(argv)
    try:
        if args.command=='screen-log':
            code, summary = screen_log(args.config, args.limit)
            print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
            return code
        if args.command=='iolite-export':
            from .iolite import export_iolite
            code, report = export_iolite(args.canonical_run, args.output)
            print(json.dumps({k: report[k] for k in ('status', 'version', 'canonical_mode',
                             'correction_exported', 'usable_file', 'provisional', 'provisional_file',
                             'importer_roundtrip_tested', 'gui_workflow_tested')}, ensure_ascii=False))
            return code
        if args.command=='inspect-netcdf':
            from .netcdf import inspect_netcdf
            print(json.dumps(inspect_netcdf(args.path), ensure_ascii=False, indent=2, allow_nan=False))
            return 0
        if args.command=='preview':
            from .preview import render_preview_pair
            summary = (render_preview_pair(args.run_dir, args.channel, args.name) if args.sample_label else
                       render_preview_pair(args.run_dir, args.channel, args.name, sample_label=False))
            print(json.dumps({k: summary[k] for k in ('channel', 'rate_fitted', 'constant_lag_s',
                             'baseline_rate_ppm', 'png', 'json')}, ensure_ascii=False))
            return 0
        if args.command=='wizard':
            from .wizard import collect
            config, output = collect()
            code, report = run(config, output, True, preview=args.preview,provisional=args.provisional,
                               alignment_fallback=args.alignment_fallback, preview_sample_label=args.sample_label)
        else:
            output = args.output
            code, report = run(args.config, output, args.export, args.export_mode,
                               args.ms_first_sample_time, args.reference_note, args.preview,
                               args.provisional, args.reference_rate_ppm, args.alignment_fallback, args.sample_label,
                               tolerance_s=args.tolerance_s, tolerance_um=args.tolerance_um)
        # The standard error belongs to the single-pass rate and is printed with it, never beside the rate after iteration.
        rate = _rate_summary(report)
        print(json.dumps({'status':report['status'], 'rate_ppm':report.get('rate_ppm'), 'rate_kind':rate['rate_kind'],
                          'single_pass_rate_ppm':rate['single_pass_rate_ppm'],
                          'single_pass_rate_se_ppm':rate['single_pass_rate_se_ppm'],
                          'rate_unit':'µs s⁻¹', 'correction_exported':report['correction_exported'],
                          'exported_log_diagnostic_rerun':report['exported_log_diagnostic_rerun'],
                          'usable_file':report['usable_file'], 'provisional_file':report.get('provisional_file'),
                          'output':str(Path(output).resolve()),
                          'preview':report.get('preview'), 'notes':report.get('notes', [])}, ensure_ascii=False))
        if args.command=='wizard':
            if report['correction_exported']:
                print('Verified log: '+str(report['usable_file']))
            elif report.get('provisional_file'):
                print('Provisional (not verified) log: '+str(report['provisional_file'])+'. Read report.md for the failed checks.')
            else:
                print('No usable correction. Read report.md for the failed check.')
            for note in report.get('notes', []):
                print(note)
            print(report.get('correction_note',''))
        return code
    except (InputError, ValueError, KeyError, OSError, EOFError) as exc:
        print(json.dumps({'status':'input_or_tool_error','error':str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    except Exception as exc:
        print(json.dumps({'status':'tool_error','type':type(exc).__name__,'error':str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


def _configure_utf8_stdio() -> None:
    """Use UTF-8 for console output even when the active code page cannot encode it."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, 'reconfigure', None)
        if callable(reconfigure):
            errors = getattr(stream, 'errors', None)
            if errors is None:
                reconfigure(encoding='utf-8')
            else:
                reconfigure(encoding='utf-8', errors=errors)
