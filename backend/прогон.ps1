# Прогон замера одной командой.
#
# Зачем скрипт, если это три строки. Потому что три строки надо помнить в
# правильном порядке, а сегодня уже был случай, когда ingest отработал до
# build_pdf и прочитал старые PDF. Скрипт помнит порядок за нас.
#
# Запуск из C:\App\pipewiki\backend:
#     .\прогон.ps1 агент
# Второй и последующие разы, чтобы сравнить с прошлым:
#     .\прогон.ps1 агент2 агент

param(
    [Parameter(Mandatory = $true)][string]$Метка,
    [string]$Сравнить = ""
)

$ErrorActionPreference = "Stop"

# 1. Стенд обязан отвечать. Без него двенадцать агентских вопросов упадут в
#    отказ, и прогон покажет не качество системы, а то, что сервис не поднят.
Write-Host "проверяю стенд FATIGUE-API..." -ForegroundColor DarkGray
try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:8100/health" -TimeoutSec 3
    Write-Host ("  стенд отвечает: парк {0}, скважин {1}" -f $health.fleet, $health.wells) -ForegroundColor DarkGreen
    if ($health.rigged) {
        Write-Host ("  испытательные трубы: устаревший расчёт {0}, воркер недоступен {1}" -f `
            $health.rigged.stale, $health.rigged.broken) -ForegroundColor DarkGray
    }
}
catch {
    Write-Host "  стенд НЕ отвечает на 127.0.0.1:8100" -ForegroundColor Red
    Write-Host "  подними его в отдельном окне и повтори:" -ForegroundColor Red
    Write-Host "      cd C:\App\pipewiki\mock-fatigue; python fatigue_api.py" -ForegroundColor Yellow
    exit 1
}

# 2. Разметка обязана быть достижима на текущем индексе. Набор, в котором
#    ожидаемая подстрока не встречается в корпусе, измеряет наши опечатки.
Write-Host "проверяю разметку набора..." -ForegroundColor DarkGray
python scripts\eval.py validate | Select-String -Pattern "проблем разметки"

# 3. Сам прогон.
python scripts\eval.py answer --label $Метка
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# 4. Сравнение, если есть с чем. Порог значимости — из последнего замера
#    разброса; без него любая разница в два вопроса объявляется значимой.
if ($Сравнить -ne "") {
    Write-Host ""
    python scripts\eval.py compare $Сравнить $Метка --noise 0.02
}

Write-Host ""
Write-Host "готово. Файл прогона — в var\eval\runs\, его и присылай." -ForegroundColor Green
