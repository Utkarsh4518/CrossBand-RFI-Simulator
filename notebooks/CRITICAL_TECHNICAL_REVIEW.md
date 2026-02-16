# Critical Technical Review: RFI Extended Analysis Model and Results

## 1. Assumptions That Could Significantly Bias Results

### 1.1 Cross-band robustness ranking

- **Identical geometry for all bands.** The same LEO orbit (circular equatorial, 1200 km, fixed initial phase) and the same GEO location (longitude 0°) are used for every band. Slant range and off-axis angle time series are therefore identical. Real systems see different elevation statistics and pass geometries depending on orbit and site; locking geometry across bands makes the ranking purely a function of victim link and antenna pattern, not of how often each band is actually illuminated by the interferer. Ranking is thus conditional on this single geometry and may not hold for other orbits or GEO longitudes.

- **Single interferer EIRP (30 dBW) for all bands.** Cross-band comparison assumes one LEO with 30 dBW in every band. In reality, LEO systems may use different EIRPs per band (e.g. different payloads or power sharing). If higher bands typically had lower LEO EIRP, the current ranking could invert or compress.

- **Same rain scenario for all bands.** Rain probability (0.3), rate (25 mm/hr), and effective path length (5 km default) are identical. Rain time series is one draw for the full duration (either “rain” or “no rain” for the whole run). So all bands see the same binary rain state. The ranking is therefore for one realization of rain, not for a statistically representative rain climate. Changing the draw (e.g. seed) or rain parameters could change relative ordering, especially where rain attenuation dominates (higher bands).

- **Single EPFD limit (−150 dBW/m²/MHz) for all bands.** Compliance uses a placeholder limit applied uniformly. Regulatory limits are band- and service-dependent (e.g. RR Article 22, Appendix 8). Because EPFD as computed is EIRP − L_fs + G_rx − 10 log₁₀(B), higher G_rx (e.g. Ka) yields higher EPFD for the same geometry. A band-specific limit could make a band “compliant” or “non-compliant” in a way that changes the relative weight of the EPFD term in RRI and thus the ranking.

- **Equal RRI weights (0.25 each).** Throughput degradation, link availability, EPFD exceedance, and joint outage are weighted equally. No justification is given. If regulatory weight were higher on EPFD, or operational weight on availability, the order Ka > K > Ku > X > S could shift.

### 1.2 Rain impact comparison

- **Rain CDF plot is confounded.** The “Rain CDF Comparison” uses (1) `no_rain_result`: X-band, interferer 30 dBW, rain_rate = 0, rain_probability = 0; (2) “Rain + RFI”: `sixg_x_result`: X-band, interferer **40 dBW**, rain 25 mm/hr, probability 0.3. The two curves therefore differ in both **rain** and **interferer EIRP**. The CDF shift cannot be attributed to rain alone; it mixes “no rain + baseline interferer” with “rain + 6G-level interferer.” Any conclusion about “rain amplification” from this plot is not scientifically defensible without a like-for-like comparison (e.g. same 30 dBW interferer with and without rain).

- **Binary, full-duration rain.** `generate_rain_time_series` makes one random draw: with probability `rain_probability`, the entire duration is at constant `rain_rate_mm_per_hr`; otherwise zero. There is no time variability, no correlation with geometry (e.g. low elevation), and no spatial structure. Rain impact is therefore averaged over one coarse realization; percentiles of “rain impact” across many runs or climates are not available.

- **Fixed effective rain path (5 km).** `effective_rain_path_km` is constant for all bands and all elevations. ITU-R P.618 uses elevation- and frequency-dependent effective path length through the rain cell. At low elevation, path length is larger and attenuation is higher; at high elevation (e.g. GEO link), the path is shorter. Using 5 km everywhere ignores elevation and can over- or under-estimate rain attenuation depending on scenario, and distorts cross-band comparison because the same physical link would have different effective paths at different frequencies in a proper model.

### 1.3 6G X-band what-if conclusions

- **Only EIRP is varied (30 vs 40 dBW).** Geometry (orbit, duration, time step), band (X), and rain (25 mm/hr, 0.3) are identical. The “6G” scenario is thus “same deployment, higher EIRP.” Real 6G or dense terrestrial/LEO deployments would also change number of interferers, elevation distribution, and possibly duty cycle and antenna patterns. Conclusions are strictly “sensitivity to a single LEO’s EIRP,” not “6G deployment impact” in a broader sense.

- **Same orbit and GEO location.** A different LEO orbit (e.g. different inclination or altitude) or a different GEO longitude would change slant range and off-axis statistics and could change the delta between 30 and 40 dBW (e.g. if most of the time the LEO is in the sidelobes, EIRP sensitivity is different than when it is in the main lobe). The reported RRI difference (0.088) is therefore scenario-specific.

## 2. Physical Intuitiveness of Ka > K > Ku > X > S

- **Antenna gain and beamwidth.** Victim bands use G_rx (dB) and θ₃dB (deg): S 32 dB / 2.5°, X 38 / 1.5°, Ku 42 / 1.2°, K 45 / 1.0°, Ka 48 / 0.8°. Off-axis gain uses a parabolic main lobe (G_max − 12(θ/θ₃dB)²) and a floor at G_max − 30 dB. With **identical** slant range and off-axis time series for all bands, higher frequency implies a narrower main lobe, so the LEO is in the main lobe for a **smaller fraction of time** and in sidelobes (G_rx − 30) more often. So average received interference I tends to be **lower** for Ka than for S. That is consistent with Ka being more “robust” in this geometry.

- **Free-space path loss (LEO → GEO).** FSPL scales as 20 log₁₀(f) + 20 log₁₀(d). For the same slant range, higher frequency implies higher path loss from LEO to GEO, so less interference power at the victim. So again, higher bands (Ka, K, Ku) receive less I than X and S for the same interferer EIRP. That supports the ranking.

- **Rain model.** Rain specific attenuation in the code uses band-dependent k and α (e.g. S: k=0.0001, α=1.0; Ka: k=0.15, α=1.3). So for the same rain rate and path length, Ka suffers much higher attenuation than S. In runs **with** rain (e.g. Phase 5: rain_probability=0.3, 25 mm/hr), Ka’s carrier is reduced more than S’s. The ranking is still Ka > S because (a) geometry and FSPL favor Ka (less I), and (b) rain is only present 30% of the time in expectation and with one draw may not occur for the whole run. So the ranking is **not** inconsistent with the rain model, but it is sensitive to rain probability and path length: if rain were more frequent or path longer, Ka could drop relative to S.

- **EPFD formulation.** EPFD = EIRP − L_fs + G_rx − 10 log₁₀(B). Same EIRP and slant range ⇒ same L_fs per time step. So EPFD differs only by G_rx (and B is 1 MHz for all). Higher G_rx ⇒ higher EPFD ⇒ **more** exceedance for a common limit. So purely from EPFD, Ka would look worse than S. The fact that the overall ranking is Ka > S means the RRI is dominated by throughput/availability/outage (where C/(N+I) and narrow beam + higher FSPL favor Ka), not by the EPFD term. So the ranking is physically plausible **given** the current formulation and equal RRI weights, but it is not “obvious” from EPFD alone; it depends on the composite metric.

- **Summary.** Ka > K > Ku > X > S is **physically intuitive** under the model: narrower beams and higher FSPL reduce average interference for higher bands for the same geometry and interferer EIRP, and the composite RRI (throughput, availability, outage, EPFD) reflects that. The caveat is that a different geometry, different rain realization, or band-specific EPFD limits could change the order.

## 3. Simplifications That Weaken Credibility

- **Circular equatorial orbit, single GEO at 0°.** Geometry is 2D (equatorial plane). Real LEOs have inclination and multiple planes; GEOs are at various longitudes. Off-axis and slant-range statistics are therefore not representative of a global or regional aggregate. Conclusions are valid for this specific geometry only.

- **Rain path length.** Constant 5 km for all bands and elevations. ITU-R P.618 (and similar) use elevation-dependent effective path length; ignoring it makes rain attenuation and “rain vs no rain” comparisons less defensible for regulatory or planning use.

- **Rain time series.** One draw for the whole duration; no temporal or spatial correlation. Real rain has structure (cell size, duration, movement). Percentiles of performance under “rain” are not estimated; only one binary realization is used per run.

- **EPFD limit.** −150 dBW/m²/MHz is a placeholder. Real limits (e.g. RR Article 22) depend on band, service, and sharing scenario. Using a single value for all bands and declaring “compliant” or “non-compliant” is not defensible for regulatory conclusions without referencing actual limits and optionally testing sensitivity.

- **Off-axis pattern.** Parabolic main lobe (12(θ/θ₃dB)²) and −30 dB floor. ITU-R S.1528 has a more detailed pattern (main lobe, transition, sidelobes). The simplified pattern affects how often the LEO is “in beam” vs “in sidelobes” and thus the average I and EPFD. Sensitivity to the pattern (e.g. sidelobe level or transition angle) is not tested.

- **Single interferer.** No aggregation over multiple LEOs or multiple beams. Real constellations add many contributing interferers; aggregate I may scale differently with EIRP and number of satellites than a single source.

- **Atmospheric losses.** DEFAULT_L_ATM_DB = 0 and DEFAULT_L_OTHER_DB = 0. Gaseous absorption (oxygen, water vapor) is frequency-dependent and non-zero above ~10 GHz. Ignoring it slightly favors higher bands (less extra loss on the interferer path than in a full model).

- **Modulation table.** Same AMC table (0, 5, 10, 15 dB thresholds) for all bands. Real systems may use different coding/modulation per band; this is a minor simplification but homogenizes “capacity” across bands.

## 4. Three High-Impact Improvements (Without Large Complexity Increase)

1. **Use ITU-R P.618 for rain (effective path length and specific attenuation).** Replace the fixed 5 km path and the simplified k/α formula with P.618 (or equivalent): compute elevation angle from the victim GEO link (e.g. from d_km and Earth radius), derive effective path length and specific attenuation from frequency and rain rate, and apply it to the carrier. This makes rain attenuation elevation- and frequency-consistent, improves cross-band and “rain vs no rain” comparisons, and is standard practice. Implementation is one function (path length + specific attenuation) called per time step or per band.

2. **Fix the Rain CDF comparison and add a proper rain-only comparison.** For the “Rain CDF” plot, compare like-for-like: same band (X), same interferer EIRP (e.g. 30 dBW), with two runs—rain_rate = 0, rain_probability = 0 vs rain_rate = 25, rain_probability = 1 (or 0.3 with multiple realizations). Use SNR loss (or joint SNR loss) from those two runs so the CDF shift is attributable to rain. Optionally add a third curve: same 30 dBW with rain, to separate “rain only” from “6G EIRP + rain.” This removes the confound and makes “rain amplification” interpretable.

3. **Document EPFD limit and run sensitivity (or use band-specific limits).** State explicitly that −150 dBW/m²/MHz is a placeholder. Either (a) reference real limits (e.g. RR Article 22) and use band/service-appropriate values where possible, or (b) keep one value but add a short sensitivity: recompute EPFD exceedance and RRI for one band (e.g. X) or all bands for 2–3 limit values (e.g. −155, −150, −145). Report how compliance and ranking change. This clarifies that conclusions are limit-dependent and improves defensibility without changing the core propagation or RRI formula.

## Addressed Limitations

The following limitations identified in this review have been addressed in the notebook. Core modeling assumptions (geometry, rain path length, EPFD formulation, RRI weights, etc.) remain unchanged; these improvements increase interpretability and scientific defensibility only.

1. **Rain CDF Confounding Fixed**
   - The Rain CDF comparison now uses identical interferer EIRP (30 dBW) for both “No Rain” and “Rain + RFI” cases.
   - This isolates rain impact from interference power changes.

2. **EPFD Limit Sensitivity Added**
   - Added sensitivity analysis for EPFD limits (−155, −150, −145 dBW/m²/MHz).
   - Demonstrates how compliance conclusions depend on threshold selection.
   - No change to compliance logic; analysis-only addition.
