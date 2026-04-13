#!/bin/bash
# Запуск веб-интерфейса для прогнозирования паводков

# cd "$(dirname "$0")"

# Активация виртуального окружения
source venv/bin/activate

# Запуск Streamlit
echo "Запуск веб-интерфейса..."
echo "Откройте в браузере: http://localhost:8501"
echo ""
streamlit run web_app.py --server.headless true
