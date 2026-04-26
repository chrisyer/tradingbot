"""Advanced MT5 order manager with market/limit/stop-limit support."""

from dataclasses import dataclass


@dataclass
class OrderSpec:
    symbol: str
    volume: float
    side: str  # "buy" or "sell"
    order_type: str  # "market", "limit", "stop_limit"
    deviation: int = 20
    stop_price: float | None = None
    limit_price: float | None = None
    magic: int = 234000
    comment: str = "RL_Agent_v2"


class MT5OrderManager:
    def __init__(self, mt5_module):
        self.mt5 = mt5_module

    def _build_base_request(self, spec: OrderSpec) -> dict:
        return {
            "symbol": spec.symbol,
            "volume": spec.volume,
            "deviation": spec.deviation,
            "magic": spec.magic,
            "comment": spec.comment,
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }

    def send_order(self, spec: OrderSpec):
        if spec.order_type == "market":
            return self._send_market(spec)
        if spec.order_type == "limit":
            return self._send_limit(spec)
        if spec.order_type == "stop_limit":
            return self._send_stop_limit(spec)
        raise ValueError(f"Unsupported order_type: {spec.order_type}")

    def _send_market(self, spec: OrderSpec):
        tick = self.mt5.symbol_info_tick(spec.symbol)
        order_side = self.mt5.ORDER_TYPE_BUY if spec.side == "buy" else self.mt5.ORDER_TYPE_SELL
        price = tick.ask if spec.side == "buy" else tick.bid

        request = self._build_base_request(spec)
        request.update(
            {
                "action": self.mt5.TRADE_ACTION_DEAL,
                "type": order_side,
                "price": price,
            }
        )
        return self.mt5.order_send(request)

    def _send_limit(self, spec: OrderSpec):
        if spec.limit_price is None:
            raise ValueError("limit_price is required for limit orders")

        order_side = self.mt5.ORDER_TYPE_BUY_LIMIT if spec.side == "buy" else self.mt5.ORDER_TYPE_SELL_LIMIT
        request = self._build_base_request(spec)
        request.update(
            {
                "action": self.mt5.TRADE_ACTION_PENDING,
                "type": order_side,
                "price": spec.limit_price,
            }
        )
        return self.mt5.order_send(request)

    def _send_stop_limit(self, spec: OrderSpec):
        if spec.stop_price is None or spec.limit_price is None:
            raise ValueError("stop_price and limit_price are required for stop-limit orders")

        order_side = self.mt5.ORDER_TYPE_BUY_STOP_LIMIT if spec.side == "buy" else self.mt5.ORDER_TYPE_SELL_STOP_LIMIT
        request = self._build_base_request(spec)
        request.update(
            {
                "action": self.mt5.TRADE_ACTION_PENDING,
                "type": order_side,
                "price": spec.stop_price,
                "stoplimit": spec.limit_price,
            }
        )
        return self.mt5.order_send(request)
