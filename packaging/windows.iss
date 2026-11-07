#ifndef BuildRoot
  #define BuildRoot "..\dist\ParaNAM Trainer"
#endif
#ifndef OutputRoot
  #define OutputRoot "..\installers"
#endif
#ifndef Flavor
  #define Flavor "CPU"
#endif
#ifndef AppVersion
  #error AppVersion must be supplied by packaging/build.py
#endif
[Setup]
AppId={{95150E04-60A8-43D8-A3C0-0FB244A1A128}
AppName=ParaNAM Trainer
AppVersion={#AppVersion}
VersionInfoVersion={#AppVersion}
AppPublisher=DLC86
DefaultDirName={localappdata}\Programs\ParaNAM Trainer
DefaultGroupName=ParaNAM Trainer
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputRoot}
OutputBaseFilename=ParaNAM-Trainer-{#AppVersion}-Windows-x64-Setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\ParaNAM Trainer.exe
LicenseFile=..\LICENSE
CloseApplications=yes
[Files]
Source: "{#BuildRoot}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
[InstallDelete]
; Replace program libraries as a unit, including upgrades from the old CUDA bundle.
Type: filesandordirs; Name: "{app}\_internal"
[Icons]
Name: "{group}\ParaNAM Trainer"; Filename: "{app}\ParaNAM Trainer.exe"
Name: "{autodesktop}\ParaNAM Trainer"; Filename: "{app}\ParaNAM Trainer.exe"; Tasks: desktopicon
[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"
Name: "nvidia"; Description: "Install NVIDIA GPU support (downloads compatible libraries; Internet required)"; GroupDescription: "Optional acceleration:"; Flags: unchecked
[Run]
Filename: "{app}\ParaNAM Trainer.exe"; Description: "Launch ParaNAM Trainer"; Flags: nowait postinstall skipifsilent

[Code]
var
  GPUProgress: TNewProgressBar;
  GPUStatus: TNewStaticText;
  GPUActive: Boolean;
  GPUCancelFile: String;
  GPUFailure: String;

procedure InitializeWizard();
begin
  GPUStatus := TNewStaticText.Create(WizardForm);
  GPUStatus.Parent := WizardForm.InstallingPage;
  GPUStatus.Left := WizardForm.ProgressGauge.Left;
  GPUStatus.Top := WizardForm.ProgressGauge.Top + WizardForm.ProgressGauge.Height + ScaleY(18);
  GPUStatus.Width := WizardForm.ProgressGauge.Width;
  GPUStatus.Height := ScaleY(36);
  GPUStatus.AutoSize := False;
  GPUStatus.WordWrap := True;
  GPUStatus.Visible := False;
  GPUProgress := TNewProgressBar.Create(WizardForm);
  GPUProgress.Parent := WizardForm.InstallingPage;
  GPUProgress.Left := GPUStatus.Left;
  GPUProgress.Top := GPUStatus.Top + GPUStatus.Height + ScaleY(4);
  GPUProgress.Width := GPUStatus.Width;
  GPUProgress.Height := WizardForm.ProgressGauge.Height;
  GPUProgress.Max := 1000;
  GPUProgress.Visible := False;
end;

procedure GPUOutput(const S: String; const Error, FirstLine: Boolean);
var
  Rest: String;
  Separator, Position: Integer;
begin
  Log(S);
  if Pos('PNGPU|', S) <> 1 then Exit;
  Rest := Copy(S, 7, Length(S));
  Separator := Pos('|', Rest);
  if Separator = 0 then Exit;
  Position := StrToIntDef(Copy(Rest, 1, Separator - 1), -1);
  GPUStatus.Caption := Copy(Rest, Separator + 1, Length(Rest));
  if Position < 0 then
    GPUProgress.Style := npbstMarquee
  else begin
    GPUProgress.Style := npbstNormal;
    GPUProgress.Position := Position;
  end;
  WizardForm.Repaint;
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  if CurPageID = wpInstalling then begin
    GPUStatus.Visible := WizardIsTaskSelected('nvidia');
    GPUProgress.Visible := GPUStatus.Visible;
    GPUStatus.Caption := 'NVIDIA support will be prepared after the application files are installed.';
  end;
  if (CurPageID = wpFinished) and (GPUFailure <> '') then
    WizardForm.FinishedLabel.Caption := WizardForm.FinishedLabel.Caption + #13#10#13#10 + GPUFailure;
end;

procedure CancelButtonClick(CurPageID: Integer; var Cancel, Confirm: Boolean);
begin
  if GPUActive then begin
    SaveStringToFile(GPUCancelFile, 'cancel', False);
    GPUStatus.Caption := 'Cancelling GPU setup... Completed downloads are kept for a later retry.';
    Cancel := False;
    Confirm := False;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Parameters: String;
  ExitCode: Integer;
  Started: Boolean;
begin
  if CurStep <> ssPostInstall then Exit;
  GPUActive := WizardIsTaskSelected('nvidia');
  GPUCancelFile := ExpandConstant('{tmp}\paranam-gpu-cancel');
  Parameters := '--configure-installation';
  if GPUActive then begin
    Parameters := Parameters + ' --with-nvidia --cancel-file "' + GPUCancelFile + '"';
    GPUStatus.Caption := 'Preparing NVIDIA GPU support...';
    GPUProgress.Style := npbstMarquee;
  end;
  WizardForm.Repaint;
  try
    Started := ExecAndLogOutput(ExpandConstant('{app}\ParaNAM Trainer.exe'),
      Parameters, '', SW_HIDE, ewWaitUntilTerminated, ExitCode, @GPUOutput);
    if (not Started) or (ExitCode <> 0) then begin
      GPUFailure := GPUStatus.Caption;
      if not Started then
        GPUFailure := 'GPU setup could not start: ' + SysErrorMessage(ExitCode);
    end;
  except
    GPUFailure := 'GPU setup failed: ' + GetExceptionMessage;
    Log(GPUFailure);
  end;
  GPUActive := False;
end;

