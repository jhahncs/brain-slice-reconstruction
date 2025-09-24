from diffusers import DDPMScheduler 
import torch
import math

def betas_for_alpha_bar(
    num_diffusion_timesteps=1000,
    max_beta=0.999,
    alpha_transform_type="piece_wise",
):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.
        alpha_transform_type (`str`, *optional*, default to `cosine`): the type of noise schedule for alpha_bar.
                     Choose from `cosine` or `exp`

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """
    if alpha_transform_type == "cosine":

        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    elif alpha_transform_type == "exp":

        def alpha_bar_fn(t):
            return math.exp(t * -12.0)
        
    elif alpha_transform_type == "piece_wise":
        def alpha_bar_fn(t):
            t = t * 1000
            if t <= 700:
                # Quadratic decrease from 1 to 0.9 between x = 0 to 700
                return 1 - 0.1 * (t / 700)**2
            else:
                # Quadratic decrease from 0.9 to 0 between x = 700 to 1000
                return 0.9 * (1 - ((t - 700) / 300)**2)

    else:
        raise ValueError(f"Unsupported alpha_tranform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)



class PiecewiseScheduler(DDPMScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.betas = betas_for_alpha_bar(
            alpha_transform_type = "piece_wise"
        )
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    '''
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator=None,
        return_dict: bool = True,
    ) :
        
        # 1. 노이즈 예측 텐서(model_output)를 커스터마이징합니다.
        # DDPMScheduler는 이 텐서를 사용하여 노이즈를 제거하므로,
        # 노이즈를 제거하고 싶지 않은 채널의 예측값을 0으로 만듭니다.
        # 이 예시에서는 4차원 텐서의 첫 번째 채널(인덱스 0)을 제외한 모든 채널의
        # 예측 노이즈를 0으로 설정합니다.
        
        # 원본 model_output의 복사본을 만들어 조작합니다.
        # 이렇게 하면 원래 예측값은 그대로 유지됩니다.
        custom_model_output = model_output.clone()
        
        # 첫 번째 채널을 제외한 나머지 채널의 예측값을 0으로 만듭니다.
        # 텐서 형태가 [Batch, Channel, Height, Width]인 경우
        #print('custom_model_output',custom_model_output[0,:3,3:])
        
        if custom_model_output.shape[1] > 1:
            custom_model_output[..., 4] = 0.0
            custom_model_output[..., 6] = 0.0
        
        # 2. 커스터마이징된 model_output을 부모 클래스의 _step_internal 메서드에 전달합니다.
        # _step_internal은 노이즈 제거 로직의 핵심을 담고 있는 내부 메서드입니다.
        # 2. Get the alpha and beta values for the current timestep.
        timestep_index = (self.timesteps == timestep).nonzero().item()
        alpha_prod_t = self.alphas_cumprod[timestep_index]
        alpha_prod_t_prev = self.alphas_cumprod[timestep_index - 1] if timestep_index > 0 else self.one
        beta_prod_t = 1 - alpha_prod_t

        # 3. Use the DPM-Solver 1st-order formula for denoising.
        # This is a common and robust denoising formula used in diffusion models.
        current_alpha_t = self.alphas[timestep_index]
        current_beta_t = self.betas[timestep_index]
        
        # Calculate the previous sample (x_t-1)
        prev_sample = (
            1 / torch.sqrt(current_alpha_t) * (sample - current_beta_t / torch.sqrt(beta_prod_t) * custom_model_output)
        )

        #print('prev_sample',prev_sample[0,:3,3:])
        # 3. 마지막으로, step 함수의 출력 형식을 맞춥니다.
        # Diffusers 라이브러리에서 사용하는 DDPMSchedulerOutput 객체를 반환합니다.
        return prev_sample
    '''
    '''
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        DDPMScheduler의 add_noise 메서드를 오버라이드하여,
        노이즈를 특정 차원에만 적용하도록 변경합니다.
        
        Args:
            original_samples (`torch.FloatTensor`): 노이즈를 추가할 원본 샘플.
            noise (`torch.FloatTensor`): 기존에 생성된 노이즈 텐서 (이 메서드 내에서 커스터마이징).
            timesteps (`torch.IntTensor`): 노이즈 스케줄을 결정하는 타임스텝.
        Returns:
            `torch.FloatTensor`: 노이즈가 추가된 샘플.
        """
        # DDPMScheduler의 add_noise는 자체적으로 노이즈를 샘플링하지 않으므로,
        # 입력으로 받은 noise 텐서를 원하는 대로 수정하여 사용합니다.

        # 1. 원본 노이즈 텐서와 동일한 형태의 제로 텐서를 생성
        custom_noise = torch.zeros_like(noise)
        
        # 2. 특정 차원에만 가우시안 노이즈 샘플링하여 할당
        # 이 예시에서는 4차원 텐서의 두 번째 차원(채널)의 첫 번째 인덱스(0)에만 노이즈를 추가합니다.
        # 텐서 형태: [Batch, Channel, Height, Width]
        # 따라서 noise[:, 0, :, :]에만 노이즈를 채웁니다.
        custom_noise[..., 3] = torch.randn_like(noise[...,3])
        custom_noise[..., 5] = torch.randn_like(noise[...,5])
        #print('custom_noise',custom_noise[0,0,3:],noise[0,0,3:])
        # 3. 노이즈 스케줄 값을 계산합니다.
        # 이는 부모 클래스인 DDPMScheduler의 로직을 그대로 사용합니다.
        timesteps = timesteps.to(self.alphas_cumprod.device)
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1.0 - self.alphas_cumprod[timesteps]) ** 0.5
        #print('sqrt_alpha_prod',sqrt_alpha_prod,original_samples[0,1,3:],sqrt_one_minus_alpha_prod,custom_noise[0,1,3:])

        # 4. 수동으로 노이즈를 주입하여 최종 결과 반환
        #torch.Size([12]) torch.Size([12, 20, 7]) torch.Size([12]) torch.Size([12, 20, 7])
        sqrt_alpha_prod = sqrt_alpha_prod[:, None, None].to(original_samples.device)
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod[:, None, None].to(original_samples.device)


        #print(sqrt_alpha_prod.shape,original_samples.shape, sqrt_one_minus_alpha_prod.shape, custom_noise.shape)
        #print(sqrt_alpha_prod.device,original_samples.device, sqrt_one_minus_alpha_prod.device, custom_noise.device)
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * custom_noise
        #print('noisy_samples',noisy_samples[0,1,3:])
        #noisy_samples = noisy_samples*(2 * torch.pi)
        #import System
        #System.exit(1)
        return noisy_samples
    '''