from cx_Freeze import setup, Executable

setup(
    name="infinitia_demo",
    version="1.0",
    description="sign language recognizing system",
    executables=[Executable("infi.py")]
)