<?php
$funnelUrl = 'https://compareserver.tailb909df.ts.net';

$uri = $_SERVER['REQUEST_URI'];
$path = parse_url($uri, PHP_URL_PATH);

// API / health proxy
if (str_starts_with($path, '/api/') || $path === '/health') {
    $targetUrl = $funnelUrl . $uri;
    $ch = curl_init($targetUrl);
    curl_setopt_array($ch, [
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_FOLLOWLOCATION => true,
        CURLOPT_TIMEOUT => 300,
        CURLOPT_HTTPHEADER => getallheaders_str(),
        CURLOPT_CUSTOMREQUEST => $_SERVER['REQUEST_METHOD'],
        CURLOPT_POSTFIELDS => file_get_contents('php://input'),
        CURLOPT_HEADERFUNCTION => function($curl, $header) {
            $trimmed = trim($header);
            if ($trimmed !== '') {
                header($trimmed, false);
            }
            return strlen($header);
        },
    ]);

    if (str_starts_with($_SERVER['HTTP_ACCEPT'] ?? '', 'text/event-stream')) {
        curl_setopt($ch, CURLOPT_WRITEFUNCTION, function($curl, $data) {
            echo $data;
            ob_flush();
            flush();
            return strlen($data);
        });
    }

    $response = curl_exec($ch);
    $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    $error = curl_error($ch);
    curl_close($ch);

    if ($error) {
        http_response_code(502);
        header('Content-Type: application/json');
        echo json_encode(['error' => 'Proxy error: ' . $error]);
        exit;
    }

    http_response_code($httpCode);
    if (!str_starts_with($_SERVER['HTTP_ACCEPT'] ?? '', 'text/event-stream')) {
        echo $response;
    }
    exit;
}

// Static files
$docRoot = __DIR__;
$filePath = $docRoot . $path;

if ($path !== '/' && is_file($filePath)) {
    $ext = pathinfo($filePath, PATHINFO_EXTENSION);
    $mimeTypes = [
        'css' => 'text/css',
        'js' => 'application/javascript',
        'svg' => 'image/svg+xml',
        'png' => 'image/png',
        'jpg' => 'image/jpeg',
        'ico' => 'image/x-icon',
        'json' => 'application/json',
        'woff2' => 'font/woff2',
        'ttf' => 'font/ttf',
    ];
    if (isset($mimeTypes[$ext])) {
        header('Content-Type: ' . $mimeTypes[$ext]);
    }
    readfile($filePath);
    exit;
}

// SPA fallback
readfile(__DIR__ . '/index.html.bak');

function getallheaders_str() {
    $headers = [];
    foreach ($_SERVER as $name => $value) {
        if (str_starts_with($name, 'HTTP_')) {
            $headerName = str_replace('_', '-', substr($name, 5));
            $headers[] = "$headerName: $value";
        }
    }
    // Fix Host header
    $headers[] = 'Host: ' . parse_url('https://compareserver.tailb909df.ts.net', PHP_URL_HOST);
    return $headers;
}
