# Сборка фронтенда внутри образа: раньше Dockerfile копировал готовый dist/,
# которого нет в git. На чистом клоне (и в CI) сборка падала, а после правок
# в src/ уезжал устаревший бандл - новая вкладка просто не появлялась.
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
RUN npm run build

FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
