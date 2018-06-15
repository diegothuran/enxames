import numpy as np
from scipy.stats import logistic
import copy

class Particulas():
    pass

class PSO():

    def __init__(self, iteracoes, numero_particulas, inercia_inicial, c1, c2, crit_parada, min, max,
                 n_dimentions, function):

        '''

        :param iteracoes:
        :param numero_particulas:
        :param inercia_inicial:
        :param c1:
        :param c2:
        :param crit_parada:
        :param min:
        :param max:
        :param n_dimentions:
        :param function:
        '''

        self.iteracoes = iteracoes
        self.numero_particulas = numero_particulas
        self.numero_dimensoes = n_dimentions
        self.inercia_inicial = inercia_inicial
        self.c1_fixo = c1
        self.c2_fixo = c2
        self.crit_parada = crit_parada
        self.particulas = []
        self.gbest = []
        self.min = min
        self.max = max
        self.function = function
        self.particulas = [self.Criar_Particula() for i in range(self.numero_particulas)]
        self.gbest = self.particulas[0]

    def Criar_Particula(self):
        '''
        Metodo para criar e inicializar todos os compenentes das particulas do enxame
        '''
        p = Particulas()
        p.dimensao = np.array([np.random.uniform(self.min, self.max) for i in range(self.numero_dimensoes)])
        p.fitness = self.Funcao(p.dimensao)
        p.velocidade = np.array([0.0 for j in range(self.numero_dimensoes)])
        p.best = p.dimensao
        p.fit_best = p.fitness
        p.c1 = self.c1_fixo
        p.c2 = self.c2_fixo
        p.inercia = self.inercia_inicial
        self.gbest = self.particulas[0]
        return p

    def Funcao(self, posicao):
        '''
        metodo para computar o fitness de acordo com a posicao passada
        :param: posicao: vetor de float, contendo as posicoes das particulas
        :return: retorna o fitness da particula correspondente a posicao passada
        '''
        # return dp.kursawe(posicao)
        # return dp.sphere(posicao)
        # return dp.ackley(posicao)
        # return dp.bohachevsky(posicao)
        return self.function(posicao)
        # return dp.zdt6(posicao)

    def Fitness(self):
        '''
        metodo para computar o fitness de todas as particulas do enxame
        '''

        # tempo = time.time()

        for i in self.particulas:
            i.fitness = self.Funcao(i.dimensao)

    def Atualizar_Particulas(self):


        for i in self.particulas:
            for j in range(len(i.dimensao)):
                sigmoid = logistic(i.dimensao[j])

                rand = np.random.uniform()

                if rand >= sigmoid:
                    i.dimensao[j] = 1
                else:
                    i.dimensao[j] = 0

    def Pbest(self):
        '''
        Metodo para atualizar a melhor posicao atual de cada particula
        '''

        for i in self.particulas:
            if (i.fit_best >= i.fitness):
                i.best = i.dimensao
                i.fit_best = i.fitness

    def Gbest(self):
        '''
        Metodo para atualizar a melhor particula do enxame
        '''

        for i in self.particulas:
            if (i.fitness <= self.gbest.fitness):
                self.gbest = copy.deepcopy(i)