"""
Проверка готовности проекта к запуску
"""
import sys
from pathlib import Path


def check_python_version():
    """Проверка версии Python"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python версия должна быть >= 3.8")
        print(f"   Текущая версия: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✓ Python версия: {version.major}.{version.minor}.{version.micro}")
    return True


def check_packages():
    """Проверка установленных пакетов"""
    required_packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('scikit-learn', 'sklearn'),
        ('catboost', 'catboost'),
        ('lightgbm', 'lightgbm'),
        ('xgboost', 'xgboost'),
        ('shap', 'shap'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('plotly', 'plotly'),
        ('fastapi', 'fastapi'),
        ('uvicorn', 'uvicorn'),
        ('streamlit', 'streamlit'),
        ('joblib', 'joblib')
    ]
    
    missing_packages = []
    
    for display_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✓ {display_name}")
        except ImportError:
            print(f"❌ {display_name} - не установлен")
            missing_packages.append(display_name)
    
    if missing_packages:
        print(f"\n⚠️  Отсутствующие пакеты: {', '.join(missing_packages)}")
        print("Установите их: pip install -r requirements.txt")
        return False
    
    return True


def check_project_structure():
    """Проверка структуры проекта"""
    required_dirs = [
        'data', 'src', 'src/utils', 'src/features', 
        'src/models', 'models', 'reports', 'notebooks'
    ]
    
    base_path = Path('/usr/src/forte')
    
    all_ok = True
    for dir_name in required_dirs:
        dir_path = base_path / dir_name
        if dir_path.exists():
            print(f"✓ {dir_name}/")
        else:
            print(f"❌ {dir_name}/ - отсутствует")
            all_ok = False
    
    return all_ok


def check_data_files():
    """Проверка наличия файлов с данными"""
    data_dir = Path('/usr/src/forte/data')
    
    required_files = [
        'транзакции в Мобильном интернет Банкинге.csv',
        'поведенческие паттерны клиентов.csv'
    ]
    
    all_ok = True
    for file_name in required_files:
        file_path = data_dir / file_name
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"✓ {file_name} ({size_mb:.2f} MB)")
        else:
            print(f"❌ {file_name} - не найден")
            all_ok = False
    
    if not all_ok:
        print("\n⚠️  Поместите CSV файлы в директорию data/")
    
    return all_ok


def main():
    """Основная функция проверки"""
    print("="*70)
    print("ПРОВЕРКА ГОТОВНОСТИ FRAUD DETECTION SYSTEM")
    print("="*70)
    
    print("\n1. Проверка версии Python...")
    python_ok = check_python_version()
    
    print("\n2. Проверка установленных пакетов...")
    packages_ok = check_packages()
    
    print("\n3. Проверка структуры проекта...")
    structure_ok = check_project_structure()
    
    print("\n4. Проверка файлов с данными...")
    data_ok = check_data_files()
    
    print("\n" + "="*70)
    
    if python_ok and packages_ok and structure_ok:
        print("✅ СИСТЕМА ГОТОВА К РАБОТЕ!")
        
        if not data_ok:
            print("\n⚠️  Внимание: данные не загружены")
            print("   Поместите CSV файлы в data/ и запустите train.py")
        else:
            print("\n🚀 Следующий шаг: python train.py")
    else:
        print("❌ ОБНАРУЖЕНЫ ПРОБЛЕМЫ")
        print("   Исправьте ошибки выше и запустите проверку снова")
    
    print("="*70)


if __name__ == '__main__':
    main()
