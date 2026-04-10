from clip_setup import (
    compute_gate_sparsity, train_epoch, evaluate
)
from sora import prune_sora_to_lora_and_report, get_trainable_state_dict

class ModelTrainer:
    """
    Orquestração do Ciclo de Vida do Modelo.
    Esta classe gerencia a execução do treinamento, validação por época, 
    ajustes de taxa de aprendizado e a extração final dos pesos otimizados.
    """
    def __init__(self, model, train_loader, eval_loader, optimizer, sparse_optimizer, scheduler, sparse_scheduler, config, is_sora):
        """
        Inicialização de Recursos.
        Vincula o modelo aos seus respectivos carregadores de dados e otimizadores, 
        configurando também o estado inicial de esparsidade (se SoRA).
        """
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.optimizer = optimizer
        self.sparse_optimizer = sparse_optimizer
        self.scheduler = scheduler
        self.sparse_scheduler = sparse_scheduler

        self.run_mode = config["model"]["lora"]["mode"]
        self.is_sora = is_sora
        self.sora_config = config["model"].get("sora", {})

    def print_metrics(self, epoch, num_epochs, metrics, eval_acc, phase_label):
        """
        Telemetria e Diagnóstico.
        Exibe no console os indicadores de performance (Loss, Accuracy) e o 
        progresso da esparsidade estrutural (Sparsity %) em tempo real.
        """
        current_lr = self.scheduler.get_last_lr()[0]
        base_log = (
            f"[{self.run_mode}]{phase_label} Epoch {epoch + 1}/{num_epochs} - "
            f"CE: {metrics['ce_loss']:.4f} - Total: {metrics['total_loss']:.4f} - "
            f"Acc: {eval_acc * 100:.2f}% - LR: {current_lr:.6f}"
        )

        if not self.is_sora:
            print(base_log)
            return

        zeros, total_gates = compute_gate_sparsity(self.model)
        sparsity = (zeros / total_gates * 100) if total_gates > 0 else 0

        sora_log = (
            f" - Sparsity: {sparsity:.1f}% ({zeros}/{total_gates})"
            f" - λ: {self.sparse_optimizer.sparse_lambda}"
        )
        print(base_log + sora_log)

    def execute_epochs(self, num_epochs, phase_label=""):
        """
        Controle do Loop de Treinamento.
        Itera sobre o número definido de épocas, chamando as funções de treino e 
        avaliação e atualizando as agendas de taxa de aprendizado.
        """
        sparse_lambda = self.sora_config.get("sparse_lambda", 0.0) if self.is_sora else 0.0

        for epoch in range(num_epochs):
            metrics = train_epoch(self.model, self.train_loader, self.optimizer, self.sparse_optimizer, sparse_lambda)
            eval_acc = evaluate(self.model, self.eval_loader)

            self.scheduler.step()
            if self.sparse_scheduler:
                self.sparse_scheduler.step()

            self.print_metrics(epoch, num_epochs, metrics, eval_acc, phase_label)

    def finalize(self):
        """
        Consolidação do Modelo.
        Após o treino, executa a poda estrutural (se SoRA) para compactar os adaptadores 
        e retorna o estado final contendo apenas os pesos treinados (head + adapters).
        """
        if self.is_sora:
            print("\n--- Iniciando Poda Estrutural SoRA ---")
            self.model = prune_sora_to_lora_and_report(self.model)

        # Retorna apenas os pesos treinados para salvar um arquivo leve e pronto para deploy
        return get_trainable_state_dict(self.model)
