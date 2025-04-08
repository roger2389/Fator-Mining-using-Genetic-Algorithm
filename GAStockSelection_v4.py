import numpy as np
import pandas as pd
import random
import tqdm
import quantstats as qs
qs.extend_pandas()
import Tool
class GAStockSelection:
    """
    用遺傳演算法 (GA) 來解「0/1子集合」+「等權配置」+「最大化 Sharpe」的選股問題。
    
    多加一個參數 forced_n (預設 None)，若指定了數值，表示在初始化時，
    前 forced_n 條染色體會是「前 forced_n 個基因=1，其餘=0」的形狀，
    其餘 (pop_size - forced_n) 條才做隨機化。
    """

    def __init__(
        self, 
        ts_result: pd.DataFrame,
        pop_size: int = 30,
        n_generations: int = 200,
        mutation_rate: float = 0.02,
        min_pick: int = 1,
        max_pick: int = None,
        patience: int = 20,
        forced_n: int = None  # <--- 新增: 預設 None，不強制
    ):
        self.ts_result = ts_result
        self.returns_matrix = ts_result.values  # shape: (T, N)
        self.N = self.returns_matrix.shape[1]
        
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.min_pick = min_pick
        # 若 max_pick 未指定，就預設等於 N(表示不限制最多選幾檔)
        self.max_pick = self.N if (max_pick is None) else max_pick
        self.patience = patience

        # 新增: forced_n
        # 若 forced_n is not None, 表示要在初始化時先建立 forced_n 條指定形狀的染色體
        # 例如前 forced_n 基因=1，其餘=0。
        self.forced_n = forced_n

        self.best_sharpe_ = -999.0
        self.best_chromosome_ = None
    
    # ---------------------- Initialization ----------------------
    def _init_population(self):
        """
        初始化人口(產生 pop_size 個染色體)。若指定 forced_n，則:
          - 先加入 forced_n 條染色體(前 forced_n 基因=1, 其餘=0)
          - 再隨機產生 pop_size - forced_n 條
        每條染色體都需符合 min_pick/max_pick 限制。
        """
        population = []

        # 如果有指定 forced_n，需要先放入 forced_n 條"固定形狀"的染色體
        # 例: 前 forced_n 個位置=1，其餘=0
        if self.forced_n is not None:
            # 若 forced_n 本身超過 N 或不在 min_pick~max_pick 範圍內，可能要報錯
            if not (0 < self.forced_n <= self.N):
                raise ValueError(f"forced_n={self.forced_n} 不合理, 應在1~N之間.")
            
            if not (self.min_pick <= self.forced_n <= self.max_pick):
                raise ValueError(
                    f"forced_n={self.forced_n} 與 min_pick={self.min_pick}, max_pick={self.max_pick} 衝突"
                )
            
            # 要先放入 forced_n 條"固定形狀"的染色體
            # 注意: 這裡示範的是全部都一樣的形狀 (前 forced_n=1,其餘=0)
            #       如果想讓每一條 forced 染色體略有變化，可以自行修改
            forced_count = min(self.forced_n, self.pop_size) 
            # 最多也只能放 pop_size 條
            for _ in range(forced_count):
                chrom = np.zeros(self.N, dtype=int)
                chrom[:_+1] = 1
                population.append(chrom)
        
        # 接著，剩下的( pop_size - len(population) ) 條才做隨機產生
        needed_random = self.pop_size - len(population)
        for _ in range(needed_random):
            chrom = np.random.randint(0, 2, size=self.N)
            while not (self.min_pick <= chrom.sum() <= self.max_pick):
                chrom = np.random.randint(0, 2, size=self.N)
            population.append(chrom)

        return np.array(population, dtype=int)
    
    # ---------------------- Evaluate Population ----------------------
    def _evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """
        以『向量化內積』的方式，同時計算所有染色體 (pop_size 個) 的每日投組報酬，
        再算各個體的 Sharpe (Mean/Std) 當作適應度。
        
        population: shape=(pop_size, N)，每列是一個 0/1 染色體。
        回傳: shape=(pop_size,) 的陣列，對應各染色體的 Sharpe。
        """
        pop_size = population.shape[0]
        pop_df = pd.DataFrame(
            population,
            index = range(pop_size),
            columns = self.ts_result.columns
        )
        pop_sum = pop_df.sum(axis=1)
        ts_sum_all = self.ts_result.dot(pop_df.T)
        port_ret_all = ts_sum_all.div(pop_sum, axis=1)
        # pop_cagr = Tool.CAGR(port_ret_all)
        # return pop_cagr.values
        pop_means = port_ret_all.mean(axis=0)
        pop_stds  = port_ret_all.std(axis=0, ddof=1)
        pop_sharpes = pop_means / pop_stds
        invalid_mask = (pop_sum == 0) | (pop_stds == 0)
        pop_sharpes[invalid_mask] = -999.0
        return pop_sharpes.values

    # ---------------------- GA Operators ----------------------
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray):
        """單點交配 (single-point crossover)"""
        cut = np.random.randint(1, self.N)
        child1 = np.concatenate([parent1[:cut], parent2[cut:]])
        child2 = np.concatenate([parent2[:cut], parent1[cut:]])
        return child1, child2
    
    def _mutate(self, chrom: np.ndarray) -> np.ndarray:
        """以 mutation_rate 的機率翻轉 bit, 若違反 min_pick/max_pick 就棄用突變。"""
        new_chrom = chrom.copy()
        for i in range(self.N):
            if random.random() < self.mutation_rate:
                new_chrom[i] = 1 - new_chrom[i]
        if not (self.min_pick <= new_chrom.sum() <= self.max_pick):
            return chrom
        return new_chrom
    
    def _select_one(self, population: np.ndarray, fitness: np.ndarray, offset: float, total_fit: float):
        """
        輪盤式選擇(Roulette Wheel)一個個體當父母。
        fitness[i] 可能有負值 => 加 offset >= -min(fitness)
        """
        pick = random.random() * total_fit
        cum = 0.0
        for i in range(len(population)):
            cum += (fitness[i] + offset)
            if cum >= pick:
                return population[i]
        return population[-1]
    
    # ---------------------- Main GA Loop ----------------------
    def fit(self):
        """
        執行 GA 演化，直到跑完 n_generations 或「連續 patience 代沒有進步」就停止。
        """
        population = self._init_population()
        no_improve_count = 0
        
        for gen in tqdm.tqdm(range(self.n_generations)):
            fitness_array = self._evaluate_population(population)
            max_fit = np.max(fitness_array)
            #print(max_fit)
            if max_fit > self.best_sharpe_:
                self.best_sharpe_ = max_fit
                self.best_chromosome_ = population[np.argmax(fitness_array)].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= self.patience:
                break

            new_population = []
            min_f = fitness_array.min()
            offset = -min_f if min_f < 0 else 0.0
            total_fit = np.sum(fitness_array + offset)

            while len(new_population) < self.pop_size:
                p1 = self._select_one(population, fitness_array, offset, total_fit)
                p2 = self._select_one(population, fitness_array, offset, total_fit)
                child1, child2 = self._crossover(p1, p2)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                new_population.append(child1)
                new_population.append(child2)

            population = np.array(new_population[:self.pop_size])
        
        return self
    
    def get_best_solution(self):
        """
        回傳最佳解相關資訊:
        - best_chromosome: 0/1 向量
        - best_sharpe_estimated: GA 過程中記錄的最高適應度
        - chosen_stocks: 對應的股票欄位名稱
        - chosen_idx: 被選到的股票 index
        - real_sharpe: 再度計算一次實際的 Sharpe，檢驗正確性
        """
        if self.best_chromosome_ is None:
            raise ValueError("尚未執行 fit() 或尚未找到可行解。")
        
        chosen_idx = np.where(self.best_chromosome_ == 1)[0]
        chosen_stocks = self.ts_result.columns[chosen_idx]
        if len(chosen_idx) > 0:
            port_ret = self.returns_matrix[:, chosen_idx].mean(axis=1)
            real_cagr = Tool.CAGR(pd.DataFrame(port_ret,index = self.ts_result.index))
            mean_p = port_ret.mean()
            std_p = port_ret.std(ddof=1)
            real_sharpe = mean_p / std_p if std_p != 0 else np.nan
        else:
            real_sharpe = np.nan
    
        return {
            "best_chromosome": self.best_chromosome_,
            "best_sharpe_estimated": self.best_sharpe_,
            "chosen_stocks": chosen_stocks,
            "chosen_idx": chosen_idx,
            "real_sharpe": real_sharpe,
            "real_CAGR":real_cagr,
        }
