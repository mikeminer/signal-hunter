▶️ How to Run the Python File on Windows (Step-by-Step)

This guide explains how to run the ETH Liquidity Grab Signals Python file on Windows using PowerShell, even if you are not a developer.

1️⃣ Install Python (Required)

Download Python from:
👉 https://www.python.org/downloads/windows/

During installation:

✅ Check “Add Python to PATH”

✅ Choose Python 3.12 or 3.13

Verify installation:

python --version


Expected output:

Python 3.12.x   (or 3.13.x)

2️⃣ Open PowerShell

Press:

Win + X → Windows Terminal / PowerShell

3️⃣ Go to the Project Folder

Example: project is on Desktop

cd C:\Users\YOUR_USERNAME\Desktop


Check files:

dir


You should see:

signale.py


⚠️ IMPORTANT
Do NOT name the file signal.py
It conflicts with Python internal modules.

4️⃣ (Recommended) Create a Virtual Environment

This avoids errors like No module named numpy.

python -m venv venv


Activate it:

.\venv\Scripts\Activate.ps1


Prompt will change to:

(venv) PS C:\Users\...\Desktop>

5️⃣ Install Required Libraries
pip install --upgrade pip
pip install numpy pandas pyqt6 websockets


Verify:

python -c "import numpy, pandas; from PyQt6.QtCore import QObject; print('ALL OK')"

6️⃣ Run the Python File
python signale.py


✔️ The application window will open
✔️ WebSocket connects automatically
✔️ Alerts activate only on real signals

7️⃣ Stop the Program

Press STOP inside the app

Or close the window

Or press:

Ctrl + C

8️⃣ Common Errors & Fixes
❌ python is not recognized

Reinstall Python and check Add to PATH

❌ No module named numpy

You forgot to activate the venv:

.\venv\Scripts\Activate.ps1


Then run again.

  ❌ App opens and closes instantly

Run with console:

python signale.py


Read the error message.

9️⃣ Optional: Build a Windows .exe

Install PyInstaller:

pip install pyinstaller


Build EXE:

pyinstaller --onefile --windowed --collect-all numpy --collect-all pandas signale.py


Executable location:

dist\signale.exe


Run it:

.\dist\signale.exe

🔑 Rules to Remember
Action	Command
Run Python file	python signale.py
Activate venv	.\venv\Scripts\Activate.ps1
Run EXE	.\signale.exe
Stop	Ctrl + C or STOP button
🧠 Final Notes

If the app is silent → market has no edge

Alerts are rare by design

This tool gives timing, not bias

Always manage risk manually
