# Praat script for cross-correlation analysis
# This script performs cross-correlation between two audio files
# to find the optimal time alignment

# Usage: praat --run crosscorrelate.praat audio1.wav audio2.wav

# Get command line arguments
if numberOfSelectedObjects = 0
    # If no objects selected, try to get from command line
    if praatVersion >= 6000
        # Newer Praat versions
        audio1$ = "audio1.wav"
        audio2$ = "audio2.wav"
    else
        # Older Praat versions
        audio1$ = "audio1.wav"
        audio2$ = "audio2.wav"
    endif
else
    # Use selected objects
    audio1 = selected("Sound", 1)
    audio2 = selected("Sound", 2)
endif

# Load audio files if not already loaded
if !exists("audio1")
    if fileReadable(audio1$)
        Read from file... 'audio1$'
        audio1 = selected("Sound")
    else
        exitScript: "Error: Could not read audio file 1"
    endif
endif

if !exists("audio2")
    if fileReadable(audio2$)
        Read from file... 'audio2$'
        audio2 = selected("Sound")
    else
        exitScript: "Error: Could not read audio file 2"
    endif
endif

# Get sampling rates
sr1 = Get sampling frequency
sr2 = Get sampling frequency

# Check if sampling rates match
if sr1 != sr2
    # Resample audio2 to match audio1
    select audio2
    Resample... sr1 50
    audio2_resampled = selected("Sound")
    select audio2
    Remove
    audio2 = audio2_resampled
endif

# Get durations
dur1 = Get total duration
dur2 = Get total duration

# Use the shorter duration for analysis
max_duration = min(dur1, dur2)

# Trim both sounds to the same length
select audio1
Extract part... 0 max_duration rectangular 1 no
audio1_trimmed = selected("Sound")

select audio2
Extract part... 0 max_duration rectangular 1 no
audio2_trimmed = selected("Sound")

# Perform cross-correlation
select audio1_trimmed
plus audio2_trimmed
Cross-correlate... zero

# Get the cross-correlation result
cross_corr = selected("Sound")

# Find the peak (maximum correlation)
peak_time = Get time of maximum... 0 0 Parabolic
peak_value = Get maximum... 0 0 Parabolic

# Calculate offset in samples
offset_samples = peak_time * sr1

# Print results
writeInfoLine: "Cross-correlation Analysis Results"
appendInfoLine: "Peak correlation time: ", peak_time, " seconds"
appendInfoLine: "Peak correlation value: ", peak_value
appendInfoLine: "Offset in samples: ", offset_samples
appendInfoLine: "Offset in seconds: ", peak_time

# Save results to file
writeFileLine: "crosscorrelation_results.txt", "Peak correlation time: " + string$(peak_time) + " seconds"
appendFileLine: "crosscorrelation_results.txt", "Peak correlation value: " + string$(peak_value)
appendFileLine: "crosscorrelation_results.txt", "Offset in samples: " + string$(offset_samples)
appendFileLine: "crosscorrelation_results.txt", "Offset in seconds: " + string$(peak_time)

# Clean up
select audio1_trimmed
plus audio2_trimmed
plus cross_corr
Remove

# Keep original sounds
select audio1
plus audio2
