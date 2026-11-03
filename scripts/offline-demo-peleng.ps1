param(
    [string]$DatasetDir = "data\peleng-source",
    [string]$ConfigPath = "config\peleng-unreal.local.yaml"
)
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { throw "ffmpeg is not on PATH." }
if (-not (Test-Path $ConfigPath)) { throw "Missing $ConfigPath. Copy config\peleng-unreal.example.yaml first." }
if (-not (Test-Path $DatasetDir)) { throw "Dataset directory not found: $DatasetDir" }
New-Item -ItemType Directory -Force -Path "outputs" | Out-Null
docker compose up -d mediamtx
$processes = @()
try {
    for ($i = 1; $i -le 5; $i++) {
        $video = Join-Path $DatasetDir ("Way1.FinalImage.CineCameraActor{0}.mp4" -f $i)
        if (-not (Test-Path $video)) { throw "Missing video: $video" }
        $cameraId = $i - 1
        $stdout = Join-Path $repoRoot ("outputs\ffmpeg-camera{0}.out.log" -f $cameraId)
        $stderr = Join-Path $repoRoot ("outputs\ffmpeg-camera{0}.err.log" -f $cameraId)
        $url = "rtsp://127.0.0.1:8554/camera$cameraId"
        $ffmpegArgs = @("-hide_banner", "-loglevel", "warning", "-re", "-stream_loop", "-1", "-i", $video, "-c", "copy", "-rtsp_transport", "tcp", "-f", "rtsp", $url)
        $argLine = ($ffmpegArgs | ForEach-Object { if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ } }) -join " "
        $processes += Start-Process -FilePath (Get-Command ffmpeg).Source -ArgumentList $argLine -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    }
    Start-Sleep -Seconds 3
    & uv run peleng-tracker run --config $ConfigPath
}
finally {
    foreach ($proc in $processes) {
        if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
    }
    docker compose stop mediamtx | Out-Null
}
