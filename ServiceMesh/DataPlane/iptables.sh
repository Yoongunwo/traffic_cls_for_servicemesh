#!/bin/bash
PROXY_PORT=${PROXY_PORT:-8080}

# 기존 규칙 초기화
iptables -t nat -F OUTPUT
iptables -t nat -F PREROUTING

iptables -t nat -A PREROUTING -p tcp --dport 9011 -j REDIRECT --to-port 9011
iptables -t nat -A PREROUTING -p tcp ! --dport $PROXY_PORT -j REDIRECT --to-port $PROXY_PORT

# OUTPUT 체인 규칙
iptables -t nat -I OUTPUT -m owner --uid-owner proxyuser -j RETURN
iptables -t nat -I OUTPUT -s 127.0.0.1/32 -j RETURN
iptables -t nat -A OUTPUT -p tcp ! --dport $PROXY_PORT -j REDIRECT --to-port $PROXY_PORT
