; installer_script.nsi

!define APP_NAME "Catch-An-Eye"
!define EXE_NAME "CatchAnEye.exe"
!define VERSION "1.0"
!define REDIST_EXE "VC_redist.x64.exe"

OutFile "${APP_NAME}_Installer_v${VERSION}.exe"
InstallDir "$PROGRAMFILES64\${APP_NAME}"
RequestExecutionLevel admin

Page directory
Page instfiles
UninstPage uninstConfirm
UninstPage instfiles

Section "Instalación Principal" SEC_MAIN
    ; Instalar el C++ Redistributable
    SetOutPath $INSTDIR
    File "${REDIST_EXE}"
    ExecWait '"$INSTDIR\${REDIST_EXE}" /q /norestart'
    
    ; Instalar la aplicación
    SetOutPath $INSTDIR
    
    ; --- MODIFICACIÓN CLAVE AQUÍ ---
    ; En lugar de copiar solo 'dist\${EXE_NAME}', copiamos el contenido
    ; de la carpeta 'dist\CatchAnEye' a la carpeta de instalación.
    File /r "dist\CatchAnEye\*"
    
    ; Crear accesos directos y desinstalador
    CreateShortCut "$DESKTOP\${APP_NAME}.lnk" "$INSTDIR\${EXE_NAME}"
    WriteUninstaller "$INSTDIR\Uninstall.exe"
SectionEnd

Section "Uninstall" SEC_UNINSTALL
    ; Borrar los archivos y carpetas
    Delete "$INSTDIR\Uninstall.exe"
    Delete "$INSTDIR\${REDIST_EXE}"
    RMDir /r "$INSTDIR"
    
    ; Borrar el acceso directo
    Delete "$DESKTOP\${APP_NAME}.lnk"
SectionEnd