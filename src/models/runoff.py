"""
Модуль расчета стока и маршрутизации.

Включает:
- Расчет времени концентрации
- Расчет гидрографа стока
- Маршрутизацию по руслу
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
from scipy.signal import convolve
from scipy.interpolate import interp1d

from ..config import HydrologicalParams, REGIONS


@dataclass
class WatershedParameters:
    """Параметры водосборного бассейна."""
    area_km2: float              # Площадь водосбора
    main_channel_length_km: float  # Длина главного русла
    average_slope: float          # Средний уклон (м/м)
    cn: float                     # Номер кривой стока
    manning_n: float = 0.035      # Коэффициент Маннинга


class RunoffCalculator:
    """
    Калькулятор стока для водосборного бассейна.
    """

    def __init__(self, region: str = "chui"):
        """
        Инициализация калькулятора.

        Args:
            region: Код региона
        """
        self.region = region
        self.region_config = REGIONS.get(region)
        self.params = HydrologicalParams()

    def calculate_time_of_concentration(
        self,
        watershed: WatershedParameters,
        method: str = "kirpich"
    ) -> float:
        """
        Расчет времени концентрации водосбора.

        Args:
            watershed: Параметры водосбора
            method: Метод расчета ('kirpich', 'scs', 'bransby_williams')

        Returns:
            Время концентрации в часах
        """
        L = watershed.main_channel_length_km
        S = watershed.average_slope

        if method == "kirpich":
            # Формула Кирпича: Tc = 0.0078 * L^0.77 * S^(-0.385)
            # L в футах, S в футах/футы
            L_ft = L * 3280.84  # км в футы
            tc_min = 0.0078 * (L_ft ** 0.77) * (S ** (-0.385))
            tc_hours = tc_min / 60

        elif method == "scs":
            # Метод SCS: Tc = L^0.8 * ((1000/CN) - 9)^0.7 / (1900 * S^0.5)
            L_ft = L * 3280.84
            cn = watershed.cn
            tc_hours = (L_ft ** 0.8 * ((1000/cn) - 9) ** 0.7) / (1900 * (S ** 0.5)) / 60

        elif method == "bransby_williams":
            # Формула Брансби-Вильямса
            A = watershed.area_km2
            L_km = L
            tc_hours = 0.243 * (L_km * A ** 0.1) / (S ** 0.2)

        else:
            tc_hours = self.params.DEFAULT_CONCENTRATION_TIME

        return max(tc_hours, 0.1)  # Минимум 6 минут

    def calculate_lag_time(
        self,
        time_of_concentration: float,
        method: str = "scs"
    ) -> float:
        """
        Расчет времени запаздывания.

        Args:
            time_of_concentration: Время концентрации (часы)
            method: Метод расчета

        Returns:
            Время запаздывания в часах
        """
        if method == "scs":
            # Lag = 0.6 * Tc
            return 0.6 * time_of_concentration
        else:
            return 0.5 * time_of_concentration

    def scs_unit_hydrograph(
        self,
        watershed: WatershedParameters,
        duration: float = 1.0,
        time_step: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Построение единичного гидрографа SCS.

        Args:
            watershed: Параметры водосбора
            duration: Длительность избыточных осадков (часы)
            time_step: Шаг времени (часы)

        Returns:
            Кортеж (время, расход)
        """
        tc = self.calculate_time_of_concentration(watershed)
        lag = self.calculate_lag_time(tc)

        # Время пика
        tp = duration / 2 + lag

        # Пиковый расход (м³/с на 1 мм избыточных осадков)
        A = watershed.area_km2
        qp = 2.08 * A / tp

        # Время спада (обычно 1.67 * tp)
        t_base = 2.67 * tp

        # Генерация гидрографа
        time = np.arange(0, t_base * 1.5, time_step)
        discharge = np.zeros_like(time)

        for i, t in enumerate(time):
            t_ratio = t / tp

            if t <= tp:
                # Фаза подъема
                discharge[i] = qp * (t_ratio ** 2.5)
            else:
                # Фаза спада
                discharge[i] = qp * np.exp(-1.5 * (t_ratio - 1))

        return time, discharge

    def triangular_unit_hydrograph(
        self,
        watershed: WatershedParameters,
        duration: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Построение треугольного единичного гидрографа.

        Args:
            watershed: Параметры водосбора
            duration: Длительность осадков (часы)

        Returns:
            Кортеж (время, расход)
        """
        tc = self.calculate_time_of_concentration(watershed)
        lag = self.calculate_lag_time(tc)

        tp = duration / 2 + lag
        t_base = 2.67 * tp

        A = watershed.area_km2
        qp = 0.208 * A / tp  # м³/с/мм

        time = np.array([0, tp, t_base])
        discharge = np.array([0, qp, 0])

        # Интерполяция для более плавного графика
        time_fine = np.linspace(0, t_base, 100)
        f = interp1d(time, discharge, kind='linear', fill_value=0, bounds_error=False)
        discharge_fine = f(time_fine)

        return time_fine, discharge_fine

    def convolve_hydrograph(
        self,
        unit_hydrograph: np.ndarray,
        excess_precipitation: np.ndarray,
        time_step: float = 1.0
    ) -> np.ndarray:
        """
        Свертка единичного гидрографа с избыточными осадками.

        Args:
            unit_hydrograph: Единичный гидрограф
            excess_precipitation: Избыточные осадки (мм за шаг)
            time_step: Шаг времени (часы)

        Returns:
            Результирующий гидрограф (м³/с)
        """
        # Свертка
        hydrograph = convolve(excess_precipitation, unit_hydrograph, mode='full')

        return hydrograph[:len(excess_precipitation) + len(unit_hydrograph) - 1]

    def muskingum_routing(
        self,
        inflow: np.ndarray,
        K: float,
        X: float,
        dt: float
    ) -> np.ndarray:
        """
        Маршрутизация стока методом Маскингама.

        Args:
            inflow: Входящий гидрограф (м³/с)
            K: Параметр задержки (часы)
            X: Весовой параметр (0-0.5)
            dt: Шаг времени (часы)

        Returns:
            Выходящий гидрограф
        """
        # Коэффициенты Маскингама
        denominator = 2 * K * (1 - X) + dt
        C0 = (dt - 2 * K * X) / denominator
        C1 = (dt + 2 * K * X) / denominator
        C2 = (2 * K * (1 - X) - dt) / denominator

        # Расчет выходного гидрографа
        outflow = np.zeros_like(inflow)
        outflow[0] = inflow[0]

        for i in range(1, len(inflow)):
            outflow[i] = C0 * inflow[i] + C1 * inflow[i-1] + C2 * outflow[i-1]
            outflow[i] = max(0, outflow[i])  # Не может быть отрицательным

        return outflow

    def calculate_peak_discharge(
        self,
        watershed: WatershedParameters,
        precipitation: float,
        runoff: float
    ) -> float:
        """
        Расчет пикового расхода рациональным методом.

        Q = C * i * A

        Args:
            watershed: Параметры водосбора
            precipitation: Осадки (мм)
            runoff: Сток (мм)

        Returns:
            Пиковый расход (м³/с)
        """
        tc = self.calculate_time_of_concentration(watershed)

        # Интенсивность осадков (мм/час)
        intensity = precipitation / tc

        # Коэффициент стока
        C = runoff / precipitation if precipitation > 0 else 0

        # Площадь в км²
        A = watershed.area_km2

        # Расход (м³/с)
        # Q = C * i * A * 0.278 (коэффициент перевода единиц)
        Q = C * intensity * A * 0.278

        return Q

    def calculate_flood_volume(
        self,
        hydrograph: np.ndarray,
        time_step: float
    ) -> float:
        """
        Расчет объема паводка.

        Args:
            hydrograph: Гидрограф (м³/с)
            time_step: Шаг времени (часы)

        Returns:
            Объем (м³)
        """
        # Интегрирование методом трапеций
        volume = np.trapz(hydrograph, dx=time_step * 3600)  # часы в секунды
        return volume

    def calculate_flood_duration(
        self,
        hydrograph: np.ndarray,
        threshold: float,
        time_step: float
    ) -> float:
        """
        Расчет продолжительности паводка выше порога.

        Args:
            hydrograph: Гидрограф (м³/с)
            threshold: Пороговый расход (м³/с)
            time_step: Шаг времени (часы)

        Returns:
            Продолжительность (часы)
        """
        above_threshold = hydrograph > threshold
        duration = np.sum(above_threshold) * time_step
        return duration

    def generate_design_storm(
        self,
        total_precipitation: float,
        duration: float,
        pattern: str = "scs_type_ii"
    ) -> np.ndarray:
        """
        Генерация расчетного шторма.

        Args:
            total_precipitation: Общая сумма осадков (мм)
            duration: Продолжительность (часы)
            pattern: Тип распределения

        Returns:
            Массив интенсивности осадков (мм/час)
        """
        # Число временных шагов
        n_steps = int(duration)

        if pattern == "scs_type_ii":
            # Распределение SCS Type II (характерно для континентального климата)
            # Пик в середине шторма
            t = np.linspace(0, 1, n_steps)
            cumulative = np.where(
                t < 0.5,
                0.5 * (t / 0.5) ** 3,
                0.5 + 0.5 * (1 - ((1 - t) / 0.5) ** 3)
            )
            incremental = np.diff(cumulative, prepend=0)

        elif pattern == "uniform":
            # Равномерное распределение
            incremental = np.ones(n_steps) / n_steps

        elif pattern == "front_loaded":
            # Максимум в начале (характерно для ливней)
            t = np.linspace(0, 1, n_steps)
            incremental = np.exp(-3 * t)
            incremental /= incremental.sum()

        else:
            incremental = np.ones(n_steps) / n_steps

        # Масштабирование
        precipitation = incremental * total_precipitation

        return precipitation


class FloodFrequencyAnalysis:
    """
    Анализ частоты паводков.
    """

    def __init__(self):
        pass

    def gumbel_distribution(
        self,
        data: np.ndarray,
        return_periods: List[float] = [2, 5, 10, 25, 50, 100]
    ) -> Dict[float, float]:
        """
        Расчет расходов заданной обеспеченности по распределению Гумбеля.

        Args:
            data: Ряд максимальных годовых расходов
            return_periods: Периоды повторяемости (годы)

        Returns:
            Словарь {период: расход}
        """
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)

        # Параметры распределения Гумбеля
        alpha = std * np.sqrt(6) / np.pi
        u = mean - 0.5772 * alpha

        results = {}
        for T in return_periods:
            # Приведенная переменная
            y_T = -np.log(-np.log(1 - 1/T))
            # Расход
            Q_T = u + alpha * y_T
            results[T] = Q_T

        return results

    def log_pearson_iii(
        self,
        data: np.ndarray,
        return_periods: List[float] = [2, 5, 10, 25, 50, 100]
    ) -> Dict[float, float]:
        """
        Расчет по распределению лог-Пирсона III типа.

        Args:
            data: Ряд максимальных годовых расходов
            return_periods: Периоды повторяемости

        Returns:
            Словарь {период: расход}
        """
        log_data = np.log10(data[data > 0])

        mean = np.mean(log_data)
        std = np.std(log_data, ddof=1)
        n = len(log_data)

        # Коэффициент асимметрии
        cs = n * np.sum((log_data - mean) ** 3) / ((n - 1) * (n - 2) * std ** 3)

        results = {}
        for T in return_periods:
            # Стандартная нормальная переменная
            p = 1 - 1/T
            z = self._norm_ppf(p)

            # Коэффициент частоты
            K = (2/cs) * (((z - cs/6) * cs/6 + 1) ** 3 - 1)

            # Логарифм расхода
            log_Q = mean + K * std
            results[T] = 10 ** log_Q

        return results

    def _norm_ppf(self, p: float) -> float:
        """
        Приближение обратной функции нормального распределения.
        """
        # Приближение Абрамовица-Стегуна
        if p <= 0.5:
            t = np.sqrt(-2 * np.log(p))
        else:
            t = np.sqrt(-2 * np.log(1 - p))

        c0 = 2.515517
        c1 = 0.802853
        c2 = 0.010328
        d1 = 1.432788
        d2 = 0.189269
        d3 = 0.001308

        z = t - (c0 + c1*t + c2*t**2) / (1 + d1*t + d2*t**2 + d3*t**3)

        if p > 0.5:
            z = -z

        return z


def example_runoff_calculation():
    """
    Пример расчета стока.
    """
    calc = RunoffCalculator(region="chui")

    # Параметры водосбора
    watershed = WatershedParameters(
        area_km2=150,
        main_channel_length_km=25,
        average_slope=0.05,
        cn=75,
        manning_n=0.035
    )

    # Время концентрации
    tc = calc.calculate_time_of_concentration(watershed)
    print(f"Время концентрации: {tc:.2f} часов")

    # Единичный гидрограф
    time, discharge = calc.scs_unit_hydrograph(watershed)
    peak_q = max(discharge)
    print(f"Пиковый расход единичного гидрографа: {peak_q:.2f} м³/с/мм")

    return calc, watershed


if __name__ == "__main__":
    example_runoff_calculation()
