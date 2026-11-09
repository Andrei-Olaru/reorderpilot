# Make a two-voice demo sales call (WAV) with Windows' built-in text-to-speech (no extra packages).
# Usage: powershell -ExecutionPolicy Bypass -File scripts/make_demo_call.ps1 [-Out data\demo_call.wav]
# The rep is "David", the client is "Zira": the true speaker of every line is known, so diarization can be checked.
param([string]$Out = "data\demo_call.wav")
Add-Type -AssemblyName System.Speech
$full = [System.IO.Path]::GetFullPath($Out)
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SetOutputToWaveFile($full)
$lines = @(
  @("Microsoft David Desktop", "Hi, it's Sam from the wholesaler. Do you have a minute to plan your next order?"),
  @("Microsoft Zira Desktop",  "Sure, but your last delivery was late again."),
  @("Microsoft David Desktop", "Sorry about that. Let me check what is due for you this month."),
  @("Microsoft Zira Desktop",  "We are still overstocked on Christmas decorations, so no more of those for now."),
  @("Microsoft David Desktop", "Understood. Is anything running low on your shelves?"),
  @("Microsoft Zira Desktop",  "Lunch bags sell well, but your prices went up and that worries me.")
)
foreach ($l in $lines) { $s.SelectVoice($l[0]); $s.Speak($l[1]) }
$s.Dispose()
Write-Output "wrote $full ($((Get-Item $full).Length) bytes)"
