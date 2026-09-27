# Прогон замера одной командой.
#
#   .\run.ps1 agent              — прогон с меткой agent
#   .\run.ps1 agent2 agent       — прогон agent2 и сравнение с agent
#
# Зачем скрипт, если это три команды: их надо помнить в правильном порядке.
# Сегодня уже был случай, когда ingest отработал до build_pdf и прочитал
# старые PDF. Скрипт помнит порядок за нас.

param(
    [Parameter(Mandatory = $true)][string]$Label,
    [string]$Against = ""
)

$ErrorActionPreference = "Stop"

# 1. Стенд обязан отвечать. Без него двенадцать агентских вопросов упадут в
#    отказ, и прогон покажет не качество системы, а то, что сервис не поднят.
Write-Host "checking FATIGUE-API stand..." -ForegroundColor DarkGray
try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:8100/health" -TimeoutSec 3
    Write-Host ("  ok: fleet {0}, wells {1}" -f $health.fleet, $health.wells) -ForegroundColor DarkGreen
}
catch {
    Write-Host "  stand is DOWN on 127.0.0.1:8100" -ForegroundColor Red
    Write-Host "  start it in another window:" -ForegroundColor Red
    Write-Host "      cd C:\App\pipewiki\mock-fatigue; python fatigue_api.py" -ForegroundColor Yellow
    exit 1
}

# 2. Разметка обязана быть достижима на текущем индексе. Набор, в котором
#    ожидаемая подстрока не встречается в корпусе, измеряет наши опечатки.
python scripts\eval.py validate | Select-String -Pattern "проблем разметки"

# 3. Сам прогон.
# --agent — ФЛАГ, а не метка. Без него живые вопросы не позовут ни одного
# инструмента, и tools_ok будет нулём не потому, что система плоха, а
# потому, что её не спрашивали. Прогон теперь и сам это ловит, но лучше
# не доводить.
python scripts\eval.py answer --agent --label $Label
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# 4. Сравнение, если есть с чем. Порог значимости — из замера разброса;
#    без него любая разница в два вопроса объявляется значимой.
if ($Against -ne "") {
    Write-Host ""
    python scripts\eval.py compare $Against $Label --noise 0.02
}

Write-Host ""
Write-Host "done. run file is in var\eval\runs\" -ForegroundColor Green
