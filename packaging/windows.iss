#ifndef BuildRoot
  #define BuildRoot "..\dist\ParaNAM Trainer"
#endif
#ifndef OutputRoot
  #define OutputRoot "..\installers"
#endif
#ifndef Flavor
  #define Flavor "CPU"
#endif
[Setup]
AppId={{95150E04-60A8-43D8-A3C0-0FB244A1A128}
AppName=ParaNAM Trainer
AppVersion=0.1.0
AppPublisher=DLC86
DefaultDirName={localappdata}\Programs\ParaNAM Trainer
DefaultGroupName=ParaNAM Trainer
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputRoot}
OutputBaseFilename=ParaNAM-Trainer-0.1.0-Windows-x64-{#Flavor}-Setup
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\ParaNAM Trainer.exe
LicenseFile=..\LICENSE
CloseApplications=yes
[Files]
Source: "{#BuildRoot}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
[Icons]
Name: "{group}\ParaNAM Trainer"; Filename: "{app}\ParaNAM Trainer.exe"
Name: "{autodesktop}\ParaNAM Trainer"; Filename: "{app}\ParaNAM Trainer.exe"; Tasks: desktopicon
[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"
[Run]
Filename: "{app}\ParaNAM Trainer.exe"; Description: "Launch ParaNAM Trainer"; Flags: nowait postinstall skipifsilent

