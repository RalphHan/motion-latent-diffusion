docker run -itd \
--gpus '"device=0"' \
-v `pwd`:/motion-latent-diffusion \
--restart always \
-p 8019:8019 \
ralphhan/mld \
bash /motion-latent-diffusion/run_server.sh
