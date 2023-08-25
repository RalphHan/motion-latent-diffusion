cd /motion-latent-diffusion
rm deps;ln -s /deps .
rm models;ln -s /models .
/opt/conda/envs/mld/bin/uvicorn main:app --host 0.0.0.0 --port 8019 --workers 2
