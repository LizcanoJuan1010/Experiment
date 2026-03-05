$files = Get-ChildItem -Path "c:\dev\others\experiment\Experiment_LLaVA" -Include "*.py","*.sh" -Recurse
$count = 0
foreach ($f in $files) {
    $content = [System.IO.File]::ReadAllText($f.FullName)
    if ($content -match "`r`n") {
        $content = $content -replace "`r`n", "`n"
        [System.IO.File]::WriteAllText($f.FullName, $content)
        Write-Host "Fixed: $($f.Name)"
        $count++
    }
}
Write-Host "Done. Fixed $count files."
