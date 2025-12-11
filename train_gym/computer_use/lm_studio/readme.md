wget https://lmstudio.ai/download/latest/linux/x64 -O app.AppImage
chmod a+x app.AppImage
./app.AppImage --appimage-extract-and-run

### На удаленном пк

sudo apt-get install xvfb

xvfb-run ./app.AppImage

если процесс застрял или завис
```bash
ps aux | grep xvfb
```
```console
dimweb    832781  0.0  0.0 210656 68584 ?        S    дек04   0:02 Xvfb :99 -screen 0 1280x1024x24 -nolisten tcp -auth /tmp/xvfb-run.Gfw4Ja/Xauthority
dimweb   1427312  0.0  0.0   9216  2560 pts/6    S+   15:58   0:00 grep --color=auto xvfb
```
```bash
sudo kill -9 832781
```