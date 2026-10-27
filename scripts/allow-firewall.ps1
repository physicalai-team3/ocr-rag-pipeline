# 휴대폰에서 PC의 8443 포트로 들어올 수 있게 Windows 방화벽 규칙을 추가합니다.
# 관리자 권한 PowerShell에서 실행:  powershell -ExecutionPolicy Bypass -File scripts\allow-firewall.ps1
param([int]$Port = 8443)

$name = "ocr-rag-pipeline $Port"
if (Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue) {
    Write-Host "이미 있음: $name"
} else {
    New-NetFirewallRule -DisplayName $name -Direction Inbound -Protocol TCP -LocalPort $Port -Action Allow -Profile Private,Domain | Out-Null
    Write-Host "추가됨: $name (Private/Domain 네트워크)"
}
Write-Host "Wi-Fi가 '공용' 네트워크로 잡혀 있으면 설정 > 네트워크에서 '개인'으로 바꾸세요."
