import os
import glob
import random
import time
import math
import logging
import networkx as nx
import matplotlib.pyplot as plt
from functools import partial
import multiprocessing

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class WFVSProblem:
    def __init__(self, filename):
        self.graph, self.weights = self.parse_instance(filename)
        # Costruzione di una struttura di vicinanze ottimizzata (dizionario)
        self.adj = {node: set(self.graph.neighbors(node)) for node in self.graph.nodes()}
    
    def parse_instance(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        nodes = None
        edges = None
        weights = {}
        adjacency_matrix = []
        
        for line in lines:
            if line.startswith("NODES:"):
                nodes = int(line.split(":")[1].strip())
            elif line.startswith("EDGES:"):
                edges = int(line.split(":")[1].strip())
            elif line.startswith("NODE_WEIGHT_SECTION"):
                break
        
        if nodes is None or edges is None:
            raise ValueError("Formato file errato: impossibile leggere il numero di nodi o archi.")
        
        weight_start_idx = lines.index("NODE_WEIGHT_SECTION\n") + 1
        for i in range(weight_start_idx, weight_start_idx + nodes):
            node, weight = map(int, lines[i].split())
            weights[node] = weight
        
        adjacency_start_idx = weight_start_idx + nodes + 1 
        for i in range(adjacency_start_idx, len(lines)):
            row = list(map(int, lines[i].split()))
            adjacency_matrix.append(row)
        
        G = nx.Graph()
        for i in range(nodes):
            G.add_node(i + 1, weight=weights[i + 1])
            for j, val in enumerate(adjacency_matrix[i]):
                if val == 1:
                    G.add_edge(i + 1, j + 1)
        return G, weights
    
    def evaluate(self, solution):
        return sum(self.weights[v] for v in solution)
    
    def is_valid_solution(self, solution):
        G_copy = self.graph.copy()
        G_copy.remove_nodes_from(solution)
        return nx.is_forest(G_copy)


class AdvancedHGA_Solver:
    def __init__(self, problem):
        self.problem = problem
        # Pre-calcola la lista dei nodi e usa la struttura di vicinanze ottimizzata
        self.nodes = list(self.problem.graph.nodes())
        self.adj = self.problem.adj
        
        # Inizializza cache per le valutazioni di fitness
        self.fitness_cache = {}
        
        # Adattamento dei parametri in base alle proprietà del grafo
        num_nodes = self.problem.graph.number_of_nodes()
        graph_density = nx.density(self.problem.graph)
        
        if num_nodes < 50:
            self.population_size = 40
        elif num_nodes < 100:
            self.population_size = 80
        else:
            self.population_size = 160
        
        self.generations = 1000  
        
        # Tournament size adattato in funzione della popolazione
        self.tournament_size = min(7, max(3, int(self.population_size * 0.05)))
        
        # Numero di individui elitari (5% della popolazione)
        self.elite_count = max(1, int(0.05 * self.population_size))
        
        # Mutation rate iniziale adattata: maggiore per grafi più densi
        self.initial_mutation_rate = 0.15 + 0.1 * graph_density
        self.mutation_rate = self.initial_mutation_rate
        
        # Parametri per la local search
        self.ls_max_iter = min(20, 5 + num_nodes // 30)
        self.ls_initial_temp = 1.5 + 0.5 * graph_density  # Temperatura iniziale aumentata
        self.ls_cooling = 0.98  # Raffreddamento più lento
        
        # Parametri per l'adattamento della mutation rate
        self.stagnation_threshold = 50  # Generazioni senza miglioramento
        self.stagnation_counter = 0
        
        # Parametri per la diversificazione
        self.diversity_threshold = 0.2 * num_nodes  # Soglia per attivare la diversificazione
        self.diversification_rate = 0.2  # Percentuale di individui da sostituire
        
        self.best_history = []
        self.eval_count = 0

    
    def has_cycle_union_find_induced(self, remaining):
        """
        Controlla la presenza di un ciclo sull'indotto definito dai nodi in 'remaining'
        usando l'algoritmo union-find.
        Restituisce True se viene rilevato un ciclo, altrimenti False.
        """
        parent = {node: node for node in remaining}
        rank = {node: 0 for node in remaining}
        
        def find(n):
            if parent[n] != n:
                parent[n] = find(parent[n])
            return parent[n]
        
        def union(a, b):
            rootA = find(a)
            rootB = find(b)
            if rootA == rootB:
                return False
            if rank[rootA] < rank[rootB]:
                parent[rootA] = rootB
            elif rank[rootA] > rank[rootB]:
                parent[rootB] = rootA
            else:
                parent[rootB] = rootA
                rank[rootA] += 1
            return True
        
        for u in remaining:
            for v in self.adj[u]:
                if v in remaining and u < v:  
                    if not union(u, v):
                        return True
        return False

    def find_cycle_in_remaining(self, remaining):
        """
        Rileva un ciclo nell'indotto definito da 'remaining' usando:
          1. Un rapido controllo con union-find.
          2. Se presente, una DFS sull'indotto basata sul dizionario 'adj'.
        Restituisce la lista dei nodi che compongono il ciclo oppure None.
        """
        if not self.has_cycle_union_find_induced(remaining):
            return None

        visited = set()
        parent = {}
        
        def dfs(u):
            visited.add(u)
            for v in self.adj[u]:
                if v not in remaining:
                    continue
                if v not in visited:
                    parent[v] = u
                    cycle = dfs(v)
                    if cycle is not None:
                        return cycle
                elif parent.get(u, None) != v:
                    cycle = [v]
                    cur = u
                    while cur != v:
                        cycle.append(cur)
                        cur = parent[cur]
                    cycle.append(v)
                    return cycle
            return None
        
        for node in remaining:
            if node not in visited:
                cycle = dfs(node)
                if cycle is not None:
                    return cycle
        return None

    def is_valid_solution_custom(self, solution):
        remaining = set(self.nodes) - solution
        return self.find_cycle_in_remaining(remaining) is None

    def repair_solution(self, solution):
        """
        Se la soluzione non è valida, aggiunge in modo greedy il nodo "migliore" (in base a peso e grado)
        preso dal ciclo rilevato.
        """
        candidate = set(solution)
        remaining = set(self.nodes) - candidate
        if self.find_cycle_in_remaining(remaining) is None:
            return candidate
        cycle = self.find_cycle_in_remaining(remaining)
        while cycle is not None:
            node_to_add = min(cycle, key=lambda n: self.problem.weights[n] / (len(self.adj[n]) + 1))
            candidate.add(node_to_add)
            remaining = set(self.nodes) - candidate
            if self.find_cycle_in_remaining(remaining) is None:
                break
            cycle = self.find_cycle_in_remaining(remaining)
        return candidate

    def greedy_solution(self):
        """
        Costruisce una soluzione valida greedy: parte da una soluzione vuota e aggiunge iterativamente
        il nodo "migliore" (in base a peso e grado) dai cicli rilevati.
        """
        solution = set()
        remaining = set(self.nodes) - solution
        cycle = self.find_cycle_in_remaining(remaining)
        while cycle is not None:
            node_to_add = min(cycle, key=lambda n: self.problem.weights[n] / (len(self.adj[n]) + 1))
            solution.add(node_to_add)
            remaining = set(self.nodes) - solution
            cycle = self.find_cycle_in_remaining(remaining)
        return solution


    def evaluate_solution(self, solution):
        key = tuple(sorted(solution))
        if key in self.fitness_cache:
            return self.fitness_cache[key]
        self.eval_count += 1
        val = self.problem.evaluate(solution)
        self.fitness_cache[key] = val
        return val

    def initialize_population(self):
        population = []
        # Inserisce la soluzione greedy nella popolazione
        greedy_sol = self.greedy_solution()
        population.append(greedy_sol)
        # Genera individui casuali, riparandoli e migliorandoli con local search
        while len(population) < self.population_size:
            k = random.randint(1, len(self.nodes) // 2)
            candidate = set(random.sample(self.nodes, k))
            if not self.is_valid_solution_custom(candidate):
                candidate = self.repair_solution(candidate)
            candidate = self.local_search(candidate)
            population.append(candidate)
        return population

    def tournament_selection(self, population):
        selected = random.sample(population, self.tournament_size)
        return min(selected, key=lambda sol: self.evaluate_solution(sol))
    
    def crossover(self, parent1, parent2):
        """
        Crossover uniforme:
        per ogni nodo, se entrambi i genitori lo includono, lo mantiene;
        se solo uno lo include, lo mantiene con probabilità 0.5.
        Inoltre, con probabilità 0.25 viene aggiunto un nodo casuale per aumentare la diversità.
        """
        child = set()
        for node in self.nodes:
            if node in parent1 and node in parent2:
                child.add(node)
            elif node in parent1 or node in parent2:
                if random.random() < 0.5:
                    child.add(node)
        if random.random() < 0.25:
            child.add(random.choice(self.nodes))
        return child
    
    def mutate(self, solution):
        """
        Operatore di mutazione che bilancia dinamicamente l'aggiunta o la rimozione.
        La probabilità di aggiungere è proporzionale al numero di nodi non ancora presenti,
        mentre se la soluzione è grande viene favorita la rimozione.
        """
        candidate = set(solution)
        if random.random() < self.mutation_rate:
            total_nodes = len(self.nodes)
            current_size = len(candidate)
            # Probabilità di aggiungere: se la soluzione è piccola, p_add è alta; se è grande, p_add è bassa.
            p_add = (total_nodes - current_size) / total_nodes
            if random.random() < p_add:
                # Aggiunta
                nodes_not_in_solution = set(self.nodes) - candidate
                if nodes_not_in_solution:
                    candidate.add(random.choice(list(nodes_not_in_solution)))
            else:
                # Rimozione
                if candidate:
                    success = False
                    for _ in range(5):  
                        node_to_remove = random.choice(list(candidate))
                        candidate_candidate = candidate.copy()
                        candidate_candidate.remove(node_to_remove)
                        if self.is_valid_solution_custom(candidate_candidate):
                            candidate = candidate_candidate
                            success = True
                            break
                    if not success:
                        candidate = solution  
        return candidate

    def local_search(self, solution):
        """
        Local search ibrida con simulated annealing:
        prova ad aggiungere o rimuovere un nodo, accettando mosse peggiorative con probabilità exp(-delta/T).
        La scelta tra aggiunta e rimozione è bilanciata dinamicamente in base alla dimensione della soluzione.
        """
        current = set(solution)
        if not self.is_valid_solution_custom(current):
            current = self.repair_solution(current)
        current_fit = self.evaluate_solution(current)
        best = set(current)
        best_fit = current_fit
        
        T = self.ls_initial_temp
        
        for _ in range(self.ls_max_iter):
            neighbor = set(current)
            total_nodes = len(self.nodes)
            current_size = len(neighbor)
            # Calcola la probabilità di aggiungere in base alla dimensione della soluzione
            p_add = (total_nodes - current_size) / total_nodes
            if random.random() < p_add:
                neighbor.add(random.choice(self.nodes))
            else:
                if neighbor:
                    neighbor.remove(random.choice(list(neighbor)))
            if not self.is_valid_solution_custom(neighbor):
                neighbor = self.repair_solution(neighbor)
            neighbor_fit = self.evaluate_solution(neighbor)
            delta = neighbor_fit - current_fit
            if delta < 0 or random.random() < math.exp(-delta / T):
                current, current_fit = neighbor, neighbor_fit
                if current_fit < best_fit:
                    best, best_fit = current, current_fit
            T *= self.ls_cooling
        return best

    def average_diversity(self, population):
        """
        Calcola la diversità media della popolazione come distanza media (simmetrica)
        tra coppie di soluzioni.
        """
        total_diff = 0
        count = 0
        pop_size = len(population)
        for i in range(pop_size):
            for j in range(i+1, pop_size):
                total_diff += len(population[i].symmetric_difference(population[j]))
                count += 1
        return total_diff / count if count > 0 else 0

    def diversify_population(self, population):
        """
        Sostituisce una parte degli individui peggiori con nuovi candidati casuali.
        """
        num_to_replace = int(self.diversification_rate * self.population_size)
        sorted_pop = sorted(population, key=lambda sol: self.evaluate_solution(sol), reverse=True)
        new_individuals = []
        for _ in range(num_to_replace):
            k = random.randint(1, len(self.nodes) // 2)
            candidate = set(random.sample(self.nodes, k))
            if not self.is_valid_solution_custom(candidate):
                candidate = self.repair_solution(candidate)
            candidate = self.local_search(candidate)
            new_individuals.append(candidate)
        diversified = sorted_pop[num_to_replace:] + new_individuals
        return diversified

    def run(self):
        start_time = time.time()
        population = self.initialize_population()
        best_solution = min(population, key=lambda sol: self.evaluate_solution(sol))
        best_value = self.evaluate_solution(best_solution)
        best_eval_at_best = self.eval_count
        self.best_history.append(best_value)
        
        logging.info(f"Inizio Advanced HGA: Best iniziale = {best_value}")
        
        for gen in range(self.generations):
            new_population = []
            sorted_pop = sorted(population, key=lambda sol: self.evaluate_solution(sol))
            elites = sorted_pop[:self.elite_count]
            new_population.extend(elites)
            
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                child = self.local_search(child)
                child = self.repair_solution(child)
                if self.is_valid_solution_custom(child):
                    new_population.append(child)
            population = new_population
            
            current_best = min(population, key=lambda sol: self.evaluate_solution(sol))
            current_value = self.evaluate_solution(current_best)
            self.best_history.append(current_value)
            
            if current_value < best_value or (current_value == best_value and self.eval_count < best_eval_at_best):
                best_solution = current_best
                best_value = current_value
                best_eval_at_best = self.eval_count
                self.stagnation_counter = 0
                self.mutation_rate = self.initial_mutation_rate
            else:
                self.stagnation_counter += 1
                if self.stagnation_counter >= self.stagnation_threshold:
                    self.mutation_rate = min(self.mutation_rate + 0.1, 0.5)
                    self.stagnation_counter = 0
                    avg_div = self.average_diversity(population)
                    if avg_div < self.diversity_threshold:
                        population = self.diversify_population(population)
                        logging.info(f"Generazione {gen}: Diversificazione attivata (avg diversity = {avg_div:.2f})")
            
            if gen % 50 == 0 or gen == self.generations - 1:
                avg_div = self.average_diversity(population)
                logging.info(f"Generazione {gen}: Best Value = {best_value}, Mutation Rate = {self.mutation_rate:.3f}, Avg Diversity = {avg_div:.2f}, Eval Count = {self.eval_count}")
        
        end_time = time.time()
        total_time = end_time - start_time
        iterations_per_sec = self.generations / total_time if total_time > 0 else float('inf')
        
        logging.info(f"FINAL REPORT Advanced HGA:\n"
                     f"  Generazioni: {self.generations}\n"
                     f"  Tempo totale (sec): {total_time:.2f}\n"
                     f"  Best Fitness: {best_value}\n"
                     f"  Iterazioni/sec: {iterations_per_sec:.2f}\n"
                     f"  Eval Count (al best): {best_eval_at_best}\n")
        
        return best_solution, best_value, self.generations, total_time, best_eval_at_best

    def plot_convergence(self, save_path=None):
        plt.figure(figsize=(8, 5))
        plt.plot(self.best_history, label='Best Fitness per Generazione')
        plt.xlabel('Generazione')
        plt.ylabel('Fitness (Peso FVS)')
        plt.title('Convergenza Advanced HGA')
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
        plt.close()



def process_instances(instances_folder, results_folder):
    os.makedirs(results_folder, exist_ok=True)
    instance_files = glob.glob(os.path.join(instances_folder, "*.fvs"))
    
    for instance_path in instance_files:
        instance_name = os.path.splitext(os.path.basename(instance_path))[0]
        logging.info(f"Processando istanza: {instance_name}")
        
        instance_results_folder = os.path.join(results_folder, instance_name)
        os.makedirs(instance_results_folder, exist_ok=True)
        
        logger = logging.getLogger(instance_name)
        logger.setLevel(logging.INFO)
        log_file = os.path.join(instance_results_folder, "log.txt")
        if logger.hasHandlers():
            logger.handlers.clear()
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        problem = WFVSProblem(instance_path)
        
        best_overall_value = float('inf')
        best_overall_solution = None
        best_overall_eval = None
        best_total_time = None
        best_iterations_per_sec = None
        best_solver = None
        
        # Esegui 10 run indipendenti per questa istanza
        for run in range(10):
            logger.info(f"Run {run+1}/10 per istanza {instance_name}")
            solver = AdvancedHGA_Solver(problem)
            best_sol, best_val, gens, total_time, best_eval = solver.run()
            logger.info(f"Run {run+1}: Best Fitness = {best_val}, Eval Count = {best_eval}, Tempo totale = {total_time:.2f} sec, Iter/sec = {gens/total_time:.2f}")
            # Aggiornamento: a parità di fitness prendi quella con minor eval count
            if (best_val < best_overall_value) or (best_val == best_overall_value and best_eval < best_overall_eval):
                best_overall_value = best_val
                best_overall_solution = best_sol
                best_overall_eval = best_eval
                best_total_time = total_time
                best_iterations_per_sec = gens / total_time if total_time > 0 else float('inf')
                best_solver = solver
        
        plot_path = os.path.join(instance_results_folder, "convergence.png")
        best_solver.plot_convergence(save_path=plot_path)
        
        final_info = (
            f"=== Risultati Finali per {instance_name} ===\n"
            f"Best Solution: {best_overall_solution}\n"
            f"Best Fitness: {best_overall_value}\n"
            f"Objective Function Evaluations (al best): {best_overall_eval}\n"
            f"Tempo totale (sec): {best_total_time:.2f}\n"
            f"Iterazioni/sec: {best_iterations_per_sec:.2f}\n"
        )
        logger.info(final_info)
        print(final_info)

if __name__ == "__main__":
    instances_folder = "istances"
    results_folder = "results_ultimi4"
    process_instances(instances_folder, results_folder)
