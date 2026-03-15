# core/timeframe_scheduler.py - SCHEDULER ESPECÍFICO POR TIMEFRAME

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional
import logging
from dataclasses import dataclass

@dataclass
class TimeframeEvent:
    timeframe: str
    trigger_time: datetime
    candle_close_time: datetime
    symbols: List[str]

class TimeframeScheduler:
    def __init__(self, delay_seconds: int = 37):  # 35s = 30s stream + 5s análise
        self.logger = logging.getLogger(__name__)
        self.delay_seconds = delay_seconds
        self.active_timeframes = ["1h"]
        self.is_running = False
        self.scheduler_thread = None
        self.timeframe_callbacks: Dict[str, Callable] = {}
        self.stop_event = threading.Event()
        
        self.logger.info(f"🕒 TimeframeScheduler inicializado:")
        self.logger.info(f"  • Delay após fechamento: {self.delay_seconds}s (30s stream + 5s análise)")
        self.logger.info(f"  • 5m: XX:00:35, XX:05:35, XX:10:35...")
        self.logger.info(f"  • 15m: XX:00:35, XX:15:35, XX:30:35...")
    
    def register_timeframe_callback(self, timeframe: str, callback: Callable):
        self.timeframe_callbacks[timeframe] = callback
        self.logger.info(f"✅ Callback registrado para {timeframe}")
    
    def _should_trigger_now(self, timeframe: str, current_time: datetime) -> bool:
        current_minute = current_time.minute
        current_second = current_time.second
        
        if current_second != self.delay_seconds:
            return False
        
        if timeframe == "5m":
            return current_minute % 5 == 0
        elif timeframe == "15m":
            return current_minute % 15 == 0
        
        return False
    
    def _get_candle_close_time(self, timeframe: str, trigger_time: datetime) -> datetime:
        return trigger_time - timedelta(seconds=self.delay_seconds)
    
    def _scheduler_loop(self):
        self.logger.info("🚀 Scheduler iniciado - aguarda stream gravar candles")
        
        while not self.stop_event.is_set():
            try:
                current_time = datetime.now()
                
                for timeframe in self.active_timeframes:
                    if self._should_trigger_now(timeframe, current_time):
                        candle_close_time = self._get_candle_close_time(timeframe, current_time)
                        
                        event = TimeframeEvent(
                            timeframe=timeframe,
                            trigger_time=current_time,
                            candle_close_time=candle_close_time,
                            symbols=[]
                        )
                        
                        if timeframe in self.timeframe_callbacks:
                            self.logger.info(
                                f"🎯 DISPARO {timeframe}: Candle fechou às {candle_close_time.strftime('%H:%M:%S')}, "
                                f"stream gravou, análise às {current_time.strftime('%H:%M:%S')}"
                            )
                            
                            try:
                                self.timeframe_callbacks[timeframe](event)
                            except Exception as e:
                                self.logger.error(f"❌ Erro no callback {timeframe}: {e}")
                
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"❌ Erro no scheduler loop: {e}")
                time.sleep(1)
        
        self.logger.info("🛑 Scheduler parado")
    
    def start(self):
        if self.is_running:
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
            name="TimeframeScheduler"
        )
        self.scheduler_thread.start()
        
        current_time = datetime.now()
        self.logger.info("📅 Próximos disparos (aguarda stream gravar):")
        for timeframe in self.active_timeframes:
            next_trigger = self._calculate_next_trigger_time(timeframe, current_time)
            time_until = next_trigger - current_time
            minutes_until = time_until.total_seconds() / 60
            
            self.logger.info(
                f"  • {timeframe}: {next_trigger.strftime('%H:%M:%S')} "
                f"(em {minutes_until:.1f} min)"
            )
    
    def _calculate_next_trigger_time(self, timeframe: str, current_time: datetime) -> datetime:
        if timeframe == "5m":
            minutes = current_time.minute
            next_close_minute = ((minutes // 5) + 1) * 5
            
            if next_close_minute >= 60:
                next_hour = current_time.hour + 1
                next_minute = next_close_minute - 60
            else:
                next_hour = current_time.hour
                next_minute = next_close_minute
                
            next_close = current_time.replace(
                hour=next_hour if next_hour < 24 else 0,
                minute=next_minute,
                second=0,
                microsecond=0
            )
            
            if next_hour >= 24:
                next_close += timedelta(days=1)
                next_close = next_close.replace(hour=0)
        
        elif timeframe == "15m":
            minutes = current_time.minute
            next_close_minute = ((minutes // 15) + 1) * 15
            
            if next_close_minute >= 60:
                next_hour = current_time.hour + 1
                next_minute = next_close_minute - 60
            else:
                next_hour = current_time.hour
                next_minute = next_close_minute
                
            next_close = current_time.replace(
                hour=next_hour if next_hour < 24 else 0,
                minute=next_minute,
                second=0,
                microsecond=0
            )
            
            if next_hour >= 24:
                next_close += timedelta(days=1)
                next_close = next_close.replace(hour=0)
        
        trigger_time = next_close + timedelta(seconds=self.delay_seconds)
        return trigger_time
    
    def stop(self):
        if not self.is_running:
            return
        
        self.logger.info("🛑 Parando scheduler...")
        self.is_running = False
        self.stop_event.set()
        
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        
        self.logger.info("✅ Scheduler parado")
    
    def get_status(self) -> Dict:
        current_time = datetime.now()
        
        next_triggers = {}
        for timeframe in self.active_timeframes:
            next_trigger = self._calculate_next_trigger_time(timeframe, current_time)
            time_until = next_trigger - current_time
            
            next_triggers[timeframe] = {
                'next_trigger_time': next_trigger.isoformat(),
                'time_until_seconds': time_until.total_seconds(),
                'time_until_minutes': time_until.total_seconds() / 60,
                'candle_close_time': self._get_candle_close_time(timeframe, next_trigger).isoformat()
            }
        
        return {
            'status': 'running' if self.is_running else 'stopped',
            'active_timeframes': self.active_timeframes,
            'delay_seconds': self.delay_seconds,
            'stream_delay_info': '30s para stream gravar + 5s para análise',
            'registered_callbacks': list(self.timeframe_callbacks.keys()),
            'current_time': current_time.isoformat(),
            'next_triggers': next_triggers,
            'scheduler_thread_alive': self.scheduler_thread.is_alive() if self.scheduler_thread else False
        }

_global_scheduler: Optional[TimeframeScheduler] = None

def get_global_scheduler() -> TimeframeScheduler:
    global _global_scheduler
    if _global_scheduler is None:
        _global_scheduler = TimeframeScheduler()
    return _global_scheduler