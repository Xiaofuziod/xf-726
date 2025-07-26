# 选择官方nginx镜像作为基础镜像
FROM nginx:alpine

# 删除默认网页
RUN rm -rf /usr/share/nginx/html/*

# 复制当前目录的所有文件到nginx默认网页目录
COPY . /usr/share/nginx/html

# 默认暴露80端口
EXPOSE 80

# 启动nginx（镜像自带命令，无需改动）
CMD ["nginx", "-g", "daemon off;"]
