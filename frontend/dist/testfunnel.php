<?php
$ch = curl_init('https://compareserver.tailb909df.ts.net/health');
curl_setopt_array($ch, [
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_TIMEOUT => 10,
    CURLOPT_SSL_VERIFYPEER => false,
    CURLOPT_SSL_VERIFYHOST => false,
]);
$r = curl_exec($ch);
$info = curl_getinfo($ch);
$err = curl_error($ch);
curl_close($ch);
echo 'HTTP: ' . $info['http_code'] . "\n";
echo 'ERR: ' . ($err ?: 'none') . "\n";
echo 'BODY: ' . ($r ? substr($r, 0, 200) : '(empty)') . "\n";
