"""Crea Imperator.lnk con icono y AppUserModelID para la barra de tareas.

Uso:  python tools/create_shortcut.py

Genera ``C:\\Imperator\\Imperator.lnk`` apuntando a ``pythonw.exe main.py``, con
el icono ``assets/icono.ico`` y el mismo AppUserModelID que declara ``main.py``.
Requiere comtypes (solo para regenerar el acceso directo, no en runtime).
"""

import ctypes
import uuid
from ctypes import wintypes
from pathlib import Path

import comtypes
from comtypes import GUID, COMMETHOD, HRESULT

ROOT = Path(__file__).resolve().parent.parent
LNK_PATH = ROOT / "Imperator.lnk"
TARGET = ROOT / "venv" / "Scripts" / "pythonw.exe"
ICON_PATH = ROOT / "assets" / "icono.ico"
APP_USER_MODEL_ID = "imperator.plataformaia.1.0"

_PKEY_FMTID = uuid.UUID("{9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3}")
PKEY_APP_USER_MODEL_ID_PID = 5


class PROPERTYKEY(ctypes.Structure):
    _fields_ = [("fmtid", ctypes.c_byte * 16), ("pid", wintypes.DWORD)]


class PROPVARIANT(ctypes.Structure):
    _fields_ = [
        ("vt", wintypes.USHORT),
        ("wReserved1", wintypes.USHORT),
        ("wReserved2", wintypes.USHORT),
        ("wReserved3", wintypes.USHORT),
        ("pwszVal", wintypes.LPWSTR),
    ]


VT_LPWSTR = 31


def _make_propertykey(fmtid: uuid.UUID, pid: int) -> PROPERTYKEY:
    key = PROPERTYKEY()
    ctypes.memmove(key.fmtid, fmtid.bytes_le, 16)
    key.pid = pid
    return key


class IShellLinkW(comtypes.IUnknown):
    _iid_ = GUID("{000214F9-0000-0000-C000-000000000046}")
    _methods_ = [
        COMMETHOD([], HRESULT, "GetPath",
                  (["in", "out"], ctypes.c_wchar_p, "pszFile"),
                  (["in"], ctypes.c_int, "cch"),
                  (["in"], ctypes.c_void_p, "pfd"),
                  (["in"], ctypes.c_uint, "fFlags")),
        COMMETHOD([], HRESULT, "GetIDList", (["out"], ctypes.POINTER(ctypes.c_void_p), "ppidl")),
        COMMETHOD([], HRESULT, "SetIDList", (["in"], ctypes.c_void_p, "pidl")),
        COMMETHOD([], HRESULT, "GetDescription",
                  (["in", "out"], ctypes.c_wchar_p, "pszName"),
                  (["in"], ctypes.c_int, "cch")),
        COMMETHOD([], HRESULT, "SetDescription", (["in"], ctypes.c_wchar_p, "pszName")),
        COMMETHOD([], HRESULT, "GetWorkingDirectory",
                  (["in", "out"], ctypes.c_wchar_p, "pszDir"),
                  (["in"], ctypes.c_int, "cch")),
        COMMETHOD([], HRESULT, "SetWorkingDirectory", (["in"], ctypes.c_wchar_p, "pszDir")),
        COMMETHOD([], HRESULT, "GetArguments",
                  (["in", "out"], ctypes.c_wchar_p, "pszArgs"),
                  (["in"], ctypes.c_int, "cch")),
        COMMETHOD([], HRESULT, "SetArguments", (["in"], ctypes.c_wchar_p, "pszArgs")),
        COMMETHOD([], HRESULT, "GetHotkey", (["out"], ctypes.POINTER(wintypes.WORD), "pwHotkey")),
        COMMETHOD([], HRESULT, "SetHotkey", (["in"], wintypes.WORD, "wHotkey")),
        COMMETHOD([], HRESULT, "GetShowCmd", (["out"], ctypes.POINTER(ctypes.c_int), "piShowCmd")),
        COMMETHOD([], HRESULT, "SetShowCmd", (["in"], ctypes.c_int, "iShowCmd")),
        COMMETHOD([], HRESULT, "GetIconLocation",
                  (["in", "out"], ctypes.c_wchar_p, "pszIconPath"),
                  (["in"], ctypes.c_int, "cch"),
                  (["out"], ctypes.POINTER(ctypes.c_int), "piIcon")),
        COMMETHOD([], HRESULT, "SetIconLocation",
                  (["in"], ctypes.c_wchar_p, "pszIconPath"),
                  (["in"], ctypes.c_int, "iIcon")),
        COMMETHOD([], HRESULT, "SetRelativePath",
                  (["in"], ctypes.c_wchar_p, "pszPathRel"),
                  (["in"], ctypes.c_uint, "dwReserved")),
        COMMETHOD([], HRESULT, "Resolve",
                  (["in"], ctypes.c_void_p, "hwnd"),
                  (["in"], ctypes.c_uint, "fFlags")),
        COMMETHOD([], HRESULT, "SetPath", (["in"], ctypes.c_wchar_p, "pszFile")),
    ]


class IPersistFile(comtypes.IUnknown):
    _iid_ = GUID("{0000010B-0000-0000-C000-000000000046}")
    _methods_ = [
        COMMETHOD([], HRESULT, "GetClassID", (["out"], ctypes.POINTER(GUID), "pClassID")),
        COMMETHOD([], HRESULT, "IsDirty"),
        COMMETHOD([], HRESULT, "Load",
                  (["in"], ctypes.c_wchar_p, "pszFileName"),
                  (["in"], ctypes.c_uint, "dwMode")),
        COMMETHOD([], HRESULT, "Save",
                  (["in"], ctypes.c_wchar_p, "pszFileName"),
                  (["in"], ctypes.c_int, "fRemember")),
        COMMETHOD([], HRESULT, "SaveCompleted", (["in"], ctypes.c_wchar_p, "pszFileName")),
        COMMETHOD([], HRESULT, "GetCurFile", (["out"], ctypes.POINTER(ctypes.c_wchar_p), "ppszFileName")),
    ]


class IPropertyStore(comtypes.IUnknown):
    _iid_ = GUID("{886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99}")
    _methods_ = [
        COMMETHOD([], HRESULT, "GetCount", (["out"], ctypes.POINTER(wintypes.DWORD), "cProps")),
        COMMETHOD([], HRESULT, "GetAt",
                  (["in"], wintypes.DWORD, "iProp"),
                  (["out"], ctypes.POINTER(PROPERTYKEY), "pkey")),
        COMMETHOD([], HRESULT, "GetValue",
                  (["in"], ctypes.POINTER(PROPERTYKEY), "key"),
                  (["in", "out"], ctypes.POINTER(PROPVARIANT), "pv")),
        COMMETHOD([], HRESULT, "SetValue",
                  (["in"], ctypes.POINTER(PROPERTYKEY), "key"),
                  (["in"], ctypes.POINTER(PROPVARIANT), "pv")),
        COMMETHOD([], HRESULT, "Commit"),
    ]


CLSID_ShellLink = GUID("{00021401-0000-0000-C000-000000000046}")


def main() -> int:
    from comtypes import CoCreateInstance

    link = CoCreateInstance(CLSID_ShellLink, interface=IShellLinkW)
    link.SetPath(str(TARGET))
    link.SetArguments("main.py")
    link.SetWorkingDirectory(str(ROOT))
    link.SetIconLocation(str(ICON_PATH), 0)

    store = link.QueryInterface(IPropertyStore)
    key = _make_propertykey(_PKEY_FMTID, PKEY_APP_USER_MODEL_ID_PID)
    pv = PROPVARIANT()
    pv.vt = VT_LPWSTR
    pv.pwszVal = APP_USER_MODEL_ID
    store.SetValue(ctypes.byref(key), ctypes.byref(pv))
    store.Commit()

    persist = link.QueryInterface(IPersistFile)
    persist.Save(str(LNK_PATH), 1)

    print(f"OK: {LNK_PATH}")
    print(f"  -> {TARGET} main.py")
    print(f"  icono: {ICON_PATH}")
    print(f"  AUMID: {APP_USER_MODEL_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

