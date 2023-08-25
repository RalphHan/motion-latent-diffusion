cd /motion-latent-diffusion
rm deps;ln -s /data/deps .
rm models;ln -s /data/models .
/opt/conda/envs/mld/bin/uvicorn main:app --host 0.0.0.0 --port 8019 --workers 2
