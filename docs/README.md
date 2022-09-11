# OMORI Sprite GAN

This repository host source code of static web page which generate OMORI character sprite. Source code to train model is available at https://github.com/ilos-vigil/OMORI-GAN. You'll need to use HTTP or HTTPS protocol in order to load WebAssembly protocol required by ONNX library.

## Library used

* [onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web) 1.12.1
* [materializecss](https://github.com/materializecss/materialize) 1.1.0
* [jszip](https://github.com/Stuk/jszip) 3.10.1
* [FileSaver.js](https://github.com/eligrey/FileSaver.js) 2.0.4

## How to run HTTP or HTTPS server

Running HTTP server is rather simple. You could just start built-in HTTP server which is included on Python 3, then open `localhost:8080` on your browser.

```sh
python -m http.server 8080
```

Running HTTPS server locally could be tricky. One way to do it is creating self-signed certificate and run HTTPS Server with [Caddy](https://caddyserver.com/).

```sh
mkdir caddy
cd caddy
wget https://github.com/caddyserver/caddy/releases/download/v2.5.2/caddy_2.5.2_linux_amd64.tar.gz
tar -xvf caddy_2.5.2_linux_amd64.tar.gz
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -nodes -subj '/CN=localhost' -days 365
nano Caddyfile
./caddy run
```

> Content of file `Caddyfile`

```Caddyfile
{
  auto_https disable_redirects   
}

https://:8443 {
  tls cert.pem key.pem {
      protocols tls1.2 tls1.2
  }
  file_server browse {
      root ..
  }
}
```
