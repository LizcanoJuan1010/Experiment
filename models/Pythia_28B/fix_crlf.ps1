$files = Get-ChildItem 'c:\dev\others\experiment\Experiment_Pythia28B' -Recurse -Include '*.sh','*.py'
foreach ($f in $files) {
    $bytes = [System.IO.File]::ReadAllBytes($f.FullName)
    $text = [System.Text.Encoding]::UTF8.GetString($bytes)
    $cr = [char]13
    $lf = [char]10
    $crlf = "$cr$lf"
    if ($text.Contains($crlf)) {
        $text = $text.Replace($crlf, "$lf")
        [System.IO.File]::WriteAllBytes($f.FullName, [System.Text.Encoding]::UTF8.GetBytes($text))
        Write-Host "Fixed: $($f.Name)"
    }
}
Write-Host "Done"
