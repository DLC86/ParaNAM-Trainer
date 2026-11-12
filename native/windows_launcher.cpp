// Fast Win32 entry point: the Start menu need not wait for Python/Qt imports.
// Only interactive GUI launches are single-instance. Training workers, tests,
// installer setup and update helpers pass through without the GUI lock.
#define UNICODE
#define _UNICODE
#include <windows.h>
#include <cwchar>
#include <string>
#include <vector>

namespace {
constexpr wchar_t kMutexName[] = L"Local\\DLC86_ParaNAM_Trainer_GUI_7F82C1";
constexpr wchar_t kWindowTitle[] = L"ParaNAM Trainer";

BOOL CALLBACK RestoreMainWindow(HWND window, LPARAM) {
  if (!IsWindowVisible(window)) return TRUE;
  wchar_t title[256]{};
  if (GetWindowTextW(window, title, 256) > 0 &&
      wcscmp(title, kWindowTitle) == 0) {
    ShowWindow(window, SW_RESTORE);
    SetForegroundWindow(window);
    return FALSE;
  }
  return TRUE;
}
}  // namespace

int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR arguments, int) {
  std::wstring self(32768, L'\0');
  const DWORD length = GetModuleFileNameW(
      nullptr, self.data(), static_cast<DWORD>(self.size()));
  if (!length || length >= self.size()) return 1;
  self.resize(length);
  const auto separator = self.find_last_of(L"\\/");
  if (separator == std::wstring::npos) return 1;
  const std::wstring directory = self.substr(0, separator);
  const std::wstring core = directory + L"\\ParaNAM Trainer Core.exe";

  const bool interactive = arguments == nullptr || *arguments == L'\0';
  HANDLE mutex = nullptr;
  if (interactive) {
    // Acquire before starting PyInstaller: rapid second clicks will not start
    // another Python process even while the UI is still importing libraries.
    mutex = CreateMutexW(nullptr, FALSE, kMutexName);
    if (!mutex) return 1;
    if (GetLastError() == ERROR_ALREADY_EXISTS) {
      EnumWindows(RestoreMainWindow, 0);
      CloseHandle(mutex);
      return 0;
    }
  }

  std::wstring command = L"\"" + core + L"\"";
  if (!interactive) {
    command += L" ";
    command += arguments;
  }
  std::vector<wchar_t> commandBuffer(command.begin(), command.end());
  commandBuffer.push_back(L'\0');

  STARTUPINFOW start{};
  start.cb = sizeof(start);
  PROCESS_INFORMATION child{};
  if (!CreateProcessW(core.c_str(), commandBuffer.data(), nullptr, nullptr,
                      FALSE, 0, nullptr, directory.c_str(), &start, &child)) {
    MessageBoxW(nullptr, L"Could not start ParaNAM Trainer.",
                kWindowTitle, MB_OK | MB_ICONERROR);
    if (mutex) CloseHandle(mutex);
    return 1;
  }
  CloseHandle(child.hThread);
  WaitForSingleObject(child.hProcess, INFINITE);
  DWORD exitCode = 1;
  GetExitCodeProcess(child.hProcess, &exitCode);
  CloseHandle(child.hProcess);
  if (mutex) CloseHandle(mutex);
  return static_cast<int>(exitCode);
}
