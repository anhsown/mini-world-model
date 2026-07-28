$secureKey = Read-Host "Paste a NEW NVIDIA API key" -AsSecureString
$pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
try {
    $plainKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    if (-not $plainKey.StartsWith("nvapi-")) {
        throw "The key must start with nvapi-"
    }
    [Environment]::SetEnvironmentVariable("NVIDIA_API_KEY", $plainKey, "User")
    Write-Host "NVIDIA_API_KEY saved to the Windows user environment. Open a new PowerShell window."
}
finally {
    if ($pointer -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    }
    Remove-Variable plainKey -ErrorAction SilentlyContinue
}

