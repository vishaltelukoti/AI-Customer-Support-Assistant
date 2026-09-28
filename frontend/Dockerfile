FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
ARG VITE_API_BASE_URL=http://localhost:8000
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
RUN npm run build

FROM nginx:alpine
RUN addgroup -S appuser && adduser -S appuser -G appuser
COPY --from=build /app/dist /usr/share/nginx/html
RUN sed -i 's/listen       80;/listen       8080;/' /etc/nginx/conf.d/default.conf \
    && sed -i 's#pid .*;#pid /tmp/nginx.pid;#' /etc/nginx/nginx.conf \
    && chown -R appuser:appuser /var/cache/nginx /var/log/nginx /etc/nginx/conf.d
USER appuser
EXPOSE 8080
