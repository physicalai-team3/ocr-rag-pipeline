# 휴대폰에서 PC의 8443 포트로 들어올 수 있게 Windows 방화벽 규칙을 추가합니다.
# 관리자 권한 PowerShell에서 실행:  powershell -ExecutionPolicy Bypass -File scripts\allow-firewall.ps1
# 처음 붙는 Wi-Fi는 '공용'으로 잡히는 경우가 많아서 모든 프로필(Any)에 엽니다.
# 시연이 끝나면 지우기:  Remove-NetFirewallRule -DisplayName "ocr-rag-pipeline 8443"
param([int]$Port = 8443)

$name = "ocr-rag-pipeline $Port"
if (Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue) {
    # 예전 버전(Private,Domain)으로 만든 규칙도 Any로 맞춥니다
    Set-NetFirewallRule -DisplayName $name -Profile Any
    Write-Host "갱신됨: $name (모든 네트워크)"
} else {
    New-NetFirewallRule -DisplayName $name -Direction Inbound -Protocol TCP -LocalPort $Port -Action Allow -Profile Any | Out-Null
    Write-Host "추가됨: $name (모든 네트워크)"
}
