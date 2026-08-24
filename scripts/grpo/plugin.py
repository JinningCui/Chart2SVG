import asyncio
import os
import random
import re
import textwrap
from collections import Counter
from copy import deepcopy
from typing import Dict, List, Union

import json
import torch

from swift.llm import PtEngine, RequestConfig, RolloutInferRequest, Template, to_device
from swift.llm.infer.protocol import ChatCompletionResponse, ChatCompletionResponseChoice
from swift.plugin import ORM, AsyncORM, orms, rm_plugins
# register context manager(used in gym training)
from swift.plugin.context_manager import ContextManager, context_managers
from swift.plugin.env import Env, envs
from swift.plugin.multi_turn import MultiTurnScheduler, multi_turns
from swift.plugin.rm_plugin import DefaultRMPlugin
from swift.utils import get_logger

from Chart2SVG.data.semantic_tokens import svg2syntactic, syntactic2svg

logger = get_logger()
"""
TO CUSTOMIZE REWARD FUNCTION:
    Step 1: Define a Reward Class
        Implement your custom reward calculation logic within the __call__ method.
        The method accepts the model's output completions and dataset columns (passed as kwargs) as input parameters.

    Step 2: Add your reward function to the orms registry:
        orms['my_reward_function'] = MyRewardFunction

    Step 3: Configure the Arguments
        Run the script with:
        --external_plugins "Path to your plugin" \
        --reward_funcs my_reward_function
"""

import io
import multiprocessing as mp

# Defense layer 1: keep the sandbox worker at module scope for multiprocessing.
def _cairosvg_render_worker(xml_code, queue):
    try:
        import cairosvg
        # Attempt to render the SVG.
        png_data = cairosvg.svg2png(bytestring=xml_code.encode('utf-8'))
        queue.put(("success", png_data))
    except Exception as e:
        queue.put(("error", e))

class SVGDependentReward(ORM):
    """
    A single reward that enforces dependency:
      1) Syntax: must be a valid, renderable SVG
      2) Visual: only evaluated if Syntax passes; IoU + SSIM + PSNR
    """
    def __init__(self, syntax_weight: float = 0.2, visual_weight: float = 0.6, len_weight: float = 0.2, repeat_penalty_floor: float = 0.8, foreground_threshold: int = 250, crop_mode: str = "union"):
        self.syntax_weight = float(syntax_weight)
        self.visual_weight = float(visual_weight)
        self.len_weight = float(len_weight)
        self.repeat_penalty_floor = float(repeat_penalty_floor)
        self.foreground_threshold = int(foreground_threshold)
        self.crop_mode = crop_mode

    @staticmethod
    def _extract_svg(completion: str) -> str:
        if '<svg' in completion and '</svg>' in completion:
            start = completion.find('<svg')
            end = completion.rfind('</svg>') + 6
            return completion[start:end]
        
        if '[<|START_OF_SVG|>]' in completion:
            start = completion.find('[<|START_OF_SVG|>]')
            end_token = '[<|END_OF_SVG|>]'
            if end_token in completion:
                end = completion.find(end_token) + len(end_token)
                return completion[start:end]
            return completion[start:]
        return ""

    def _compute_repeat_penalty(self, svg_code: str) -> float:
        if not svg_code:
            return 1.0
        normalized = re.sub(r'\s+', ' ', svg_code).strip()
        tags = re.findall(r'<[^>]+?>', normalized)
        if not tags:
            return 1.0
        total = len(tags)
        unique = len(set(tags))
        dup_ratio = max(0.0, (total - unique) / total)
        penalty = 1.0 - (dup_ratio * 0.5)
        return max(self.repeat_penalty_floor, penalty)

    # Defense layer 2: render in a sandbox process with a hard timeout.
    @staticmethod
    def _render_svg_safe(xml_code, timeout=15.0):
        # Reject SVGs over 150,000 characters to prevent excessive memory use.
        if len(xml_code) > 150000:
            raise ValueError(f"SVG code too massive ({len(xml_code)} chars). Possible bomb.")

        # Use spawn so the worker does not inherit the DeepSpeed CUDA context.
        ctx = mp.get_context('spawn')
        q = ctx.Queue()
        p = ctx.Process(target=_cairosvg_render_worker, args=(xml_code, q))
        p.start()
        
        p.join(timeout)  # Wait for the configured timeout.
        
        if p.is_alive():
            # Terminate renderers that hang or enter an infinite loop.
            p.terminate()
            p.join()
            raise TimeoutError("CairoSVG rendering timed out. Malicious SVG killed.")
            
        if not q.empty():
            status, res = q.get()
            if status == "success":
                return res
            else:
                raise res
        raise RuntimeError("Worker process died unexpectedly.")

    @staticmethod
    def _get_content_bbox(img_arr, threshold=250):
        import numpy as np
        if img_arr.ndim == 3:
            mask = img_arr.mean(axis=2) < threshold
        else:
            mask = img_arr < threshold
        if not np.any(mask):
            return None
        coords = np.argwhere(mask)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        return (x_min, y_min, x_max, y_max)

    @staticmethod
    def _crop_to_foreground(gen_arr, gt_arr, threshold=250, mode="union"):
        gen_box = SVGDependentReward._get_content_bbox(gen_arr, threshold=threshold)
        gt_box = SVGDependentReward._get_content_bbox(gt_arr, threshold=threshold)
        if gen_box is None and gt_box is None:
            return gen_arr, gt_arr
        if gen_box is None: gen_box = gt_box
        if gt_box is None: gt_box = gen_box
        
        x_min = min(gen_box[0], gt_box[0])
        y_min = min(gen_box[1], gt_box[1])
        x_max = max(gen_box[2], gt_box[2])
        y_max = max(gen_box[3], gt_box[3])
        h, w = gen_arr.shape[:2]
        x_min = max(0, min(w - 1, x_min))
        x_max = max(0, min(w - 1, x_max))
        y_min = max(0, min(h - 1, y_min))
        y_max = max(0, min(h - 1, y_max))
        if x_max < x_min or y_max < y_min:
            return gen_arr, gt_arr
        return gen_arr[y_min:y_max + 1, x_min:x_max + 1], gt_arr[y_min:y_max + 1, x_min:x_max + 1]

    @staticmethod
    def _resize_to_match(gen_arr, gt_arr):
        import numpy as np
        from PIL import Image
        if gen_arr.shape == gt_arr.shape:
            return gen_arr, gt_arr
        h = max(gen_arr.shape[0], gt_arr.shape[0])
        w = max(gen_arr.shape[1], gt_arr.shape[1])
        gen_img = Image.fromarray(gen_arr).resize((w, h))
        gt_img = Image.fromarray(gt_arr).resize((w, h))
        return np.array(gen_img), np.array(gt_img)

    @staticmethod
    def _to_gray(arr):
        if arr.ndim == 2:
            return arr.astype('float64')
        return (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype('float64')

    @staticmethod
    def _compute_iou(gen_arr, gt_arr):
        import numpy as np
        from scipy import ndimage
        gen_gray = SVGDependentReward._to_gray(gen_arr)
        gt_gray = SVGDependentReward._to_gray(gt_arr)
        gen_mask = gen_gray < 245
        gt_mask = gt_gray < 245
        if not np.any(gen_mask) and not np.any(gt_mask): return 0.0
        structure = np.ones((5, 5))
        gen_mask_relaxed = ndimage.binary_dilation(gen_mask, structure=structure)
        gt_mask_relaxed = ndimage.binary_dilation(gt_mask, structure=structure)
        intersection = np.logical_and(gen_mask_relaxed, gt_mask).sum()
        union = np.logical_or(gen_mask, gt_mask).sum()
        if union == 0: return 0.0
        iou = float(intersection / (union + 1e-6))
        return min(max(iou, 0.0), 1.0)

    @staticmethod
    def _compute_psnr(gen_arr, gt_arr):
        import numpy as np
        gen_gray = SVGDependentReward._to_gray(gen_arr)
        gt_gray = SVGDependentReward._to_gray(gt_arr)
        mse = ((gen_gray - gt_gray) ** 2).mean()
        if mse == 0: return 1.0
        psnr = 20 * np.log10(255.0) - 10 * np.log10(mse)
        return float(min(psnr / 40.0, 1.0))

    @staticmethod
    def _compute_ssim(gen_arr, gt_arr):
        import numpy as np
        from scipy.ndimage import gaussian_filter
        gen_gray = gaussian_filter(SVGDependentReward._to_gray(gen_arr), sigma=1.5)
        gt_gray = gaussian_filter(SVGDependentReward._to_gray(gt_arr), sigma=1.5)
        mu_x = gen_gray.mean()
        mu_y = gt_gray.mean()
        sigma_x_sq = gen_gray.var()
        sigma_y_sq = gt_gray.var()
        sigma_xy = ((gen_gray - mu_x) * (gt_gray - mu_y)).mean()
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        l = (2 * mu_x * mu_y + c1) / (mu_x**2 + mu_y**2 + c1)
        cs = (2 * np.abs(sigma_xy) + c2) / (sigma_x_sq + sigma_y_sq + c2)
        ssim = l * cs
        return float(max(0.0, min(ssim, 1.0)))

    def __call__(self, completions, **kwargs) -> list[float]:
        try:
            import numpy as np
            from PIL import Image
            import io
            import os
            try:
                from Chart2SVG.data.semantic_tokens import syntactic2svg
            except ImportError:
                syntactic2svg = lambda x: x
        except ImportError as e:
            print(f"!!! CRITICAL IMPORT ERROR: {e} !!!")
            return [0.0] * len(completions)

        images = kwargs.get('images', [])
        save_dir = kwargs.get('save_dir', None)
        rewards = []

        for idx, completion in enumerate(completions):
            try:
                svg_code = self._extract_svg(completion)
                syntax_ok = False
                syntax_score = -1.0
                gen_img = None
                xml_code = ""
                
                if svg_code:
                    try:
                        if '[<|START_OF_SVG|>]' in svg_code:
                            xml_code = syntactic2svg(svg_code)
                        else:
                            xml_code = svg_code
                        
                        # Render through the sandboxed function with timeout protection.
                        png_data = self._render_svg_safe(xml_code, timeout=15.0)
                        gen_img = Image.open(io.BytesIO(png_data)).convert('RGB')
                        
                        syntax_ok = True
                        syntax_score = 1.0
                    except Exception as e:
                        # Catch both rendering timeouts and SVG syntax errors here.
                        print(f"[SVG Reward] idx={idx} syntax_render_error={type(e).__name__}: {e}")
                
                if not syntax_ok:
                    rewards.append(-1.0)
                    continue

                visual_reward = 0.0
                iou_score, ssim_score, psnr_score = 0.0, 0.0, 0.0
                
                if idx < len(images) and gen_img:
                    try:
                        gt_img_entry = images[idx]
                        if isinstance(gt_img_entry, list): gt_img_entry = gt_img_entry[0]
                        gt_img_path = None
                        
                        if isinstance(gt_img_entry, dict):
                            for key in ["path", "image", "image_path", "file", "filepath", "file_path", "url"]:
                                if key in gt_img_entry and isinstance(gt_img_entry[key], str):
                                    gt_img_path = gt_img_entry[key]
                                    break
                            if gt_img_path is None:
                                for value in gt_img_entry.values():
                                    if isinstance(value, str):
                                        gt_img_path = value
                                        break
                        else:
                            gt_img_path = gt_img_entry
                        
                        if isinstance(gt_img_path, str) and os.path.exists(gt_img_path):
                            gt_img = Image.open(gt_img_path).convert('RGB')
                            size = (512, 512)
                            gen_img = gen_img.resize(size)
                            gt_img = gt_img.resize(size)
                            gen_arr = np.array(gen_img)
                            gt_arr = np.array(gt_img)
                            
                            gen_arr, gt_arr = self._crop_to_foreground(
                                gen_arr, gt_arr, threshold=self.foreground_threshold, mode=self.crop_mode
                            )
                            gen_arr, gt_arr = self._resize_to_match(gen_arr, gt_arr)
                            
                            iou_score = self._compute_iou(gen_arr, gt_arr)
                            ssim_score = self._compute_ssim(gen_arr, gt_arr)
                            psnr_score = self._compute_psnr(gen_arr, gt_arr)
                            visual_reward = (iou_score + psnr_score + ssim_score) / 3.0
                        else:
                            print(f"[SVG Reward] idx={idx} gt_image_missing path={gt_img_path}")
                    except Exception as e:
                        print(f"[SVG Reward] idx={idx} visual_error={e}")
                
                repeat_multiplier = self._compute_repeat_penalty(xml_code)
                # total_reward = (self.syntax_weight * syntax_score) + (self.visual_weight * visual_reward) + (self.len_weight * repeat_multiplier)
                total_reward = visual_reward
                # total_reward = max(0.0, total_reward * repeat_multiplier)
                rewards.append(total_reward)
                
                print(f"[SVG Reward] idx={idx} syntax={syntax_score:.4f} visual={visual_reward:.4f} iou={iou_score:.4f} ssim={ssim_score:.4f} repeat={repeat_multiplier:.4f} total={total_reward:.4f}")
                
            except Exception as e:
                print(f"[SVG Reward] FATAL ERROR on idx={idx}: {e}")
                rewards.append(0.0)

        return rewards

orms['svg_pipeline'] = SVGDependentReward


# For additional reward functions, refer to swift/plugin/orm.py.
class CountdownORM(ORM):

    def __call__(self, completions, target, nums, **kwargs) -> List[float]:
        """
        Evaluates completions based on Mathematical correctness of the answer

        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers
            nums (list[str]): Available numbers

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        for completion, gt, numbers in zip(completions, target, nums):
            try:
                # Check if the format is correct
                match = re.search(r'<answer>(.*?)<\/answer>', completion)
                if match is None:
                    rewards.append(0.0)
                    continue
                # Extract the "answer" part from the completion
                equation = match.group(1).strip()
                if '=' in equation:
                    equation = equation.split('=')[0]
                # Extract all numbers from the equation
                used_numbers = [int(n) for n in re.findall(r'\d+', equation)]

                # Check if all numbers are used exactly once
                if sorted(used_numbers) != sorted(numbers):
                    rewards.append(0.0)
                    continue
                # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
                allowed_pattern = r'^[\d+\-*/().\s]+$'
                if not re.match(allowed_pattern, equation):
                    rewards.append(0.0)
                    continue

                # Evaluate the equation with restricted globals and locals
                result = eval(equation, {'__builtins__': None}, {})
                # Check if the equation is correct and matches the ground truth
                if abs(float(result) - float(gt)) < 1e-5:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except Exception:
                # If evaluation fails, reward is 0
                rewards.append(0.0)
        return rewards


orms['external_countdown'] = CountdownORM


class MultiModalAccuracyORM(ORM):

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        from math_verify import parse, verify
        for content, sol in zip(completions, solution):
            reward = 0.0
            # Try symbolic verification first
            try:
                answer = parse(content)
                if float(verify(answer, parse(sol))) > 0:
                    reward = 1.0
            except Exception:
                pass  # Continue to next verification method if this fails

            # If symbolic verification failed, try string matching
            if reward == 0.0:
                try:
                    # Extract answer from solution if it has think/answer tags
                    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                    # Extract answer from content if it has think/answer tags
                    content_match = re.search(r'<answer>(.*?)</answer>', content)
                    student_answer = content_match.group(1).strip() if content_match else content.strip()

                    # Compare the extracted answers
                    if student_answer == ground_truth:
                        reward = 1.0
                except Exception:
                    pass  # Keep reward as 0.0 if both methods fail
            rewards.append(reward)
        return rewards


orms['external_r1v_acc'] = MultiModalAccuracyORM


class MultiTurnThinkingTips(ORM):
    """
    A reward function example designed for use with the `ThinkingTipsScheduler`.

    This class demonstrates how to handle reward computation when a single
    training sample (or request) is split into multiple "turns" or steps.
    Specifically, it computes the reward based on the **last turn** of each
    multi-turn trajectory using a math accuracy function.

    NOTE
    ----
    If you feed fragments of the *same* trajectory as independent samples, this
    function **must return an identical reward for every fragment**
    """

    def __init__(self):
        from swift.plugin.orm import MathAccuracy
        self.acc_func = MathAccuracy()

    def __call__(self, completions, **kwargs) -> List[float]:
        trajectory_ids: List[str] = kwargs.get('request_id')

        global_trajectorys: Dict[str, List[Dict]] = kwargs.get('trajectory_inputs')

        rewards = []
        for local_tra_id in trajectory_ids:
            total_trajectory_inputs = global_trajectorys[local_tra_id]
            # For reward calculation, we use the entire trajectory of this sample.
            # Here, we specifically evaluate only the last turn.
            last_turn_messages = total_trajectory_inputs[-1]['messages']
            last_turn_completion = last_turn_messages[-1]['content']
            last_turn_solution = total_trajectory_inputs[-1]['solution']
            # Compute reward based on math accuracy for the final completion.
            reward = self.acc_func([last_turn_completion], [last_turn_solution])[0]
            rewards.append(reward)
        return rewards


orms['thinking_tips'] = MultiTurnThinkingTips


# ref implementation: https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py
class CodeReward(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('e2b') is not None, (
            "The e2b package is required but not installed. Please install it using 'pip install e2b-code-interpreter'."
        )
        from dotenv import load_dotenv
        load_dotenv()

    @staticmethod
    def extract_code(completion: str, language: str) -> str:
        pattern = re.compile(rf'```{language}\n(.*?)```', re.DOTALL)
        matches = pattern.findall(completion)
        extracted_answer = matches[-1] if len(matches) >= 1 else ''
        return extracted_answer

    def run_async_from_sync(self, scripts: List[str], languages: List[str]) -> List[float]:
        """Function wrapping the `run_async` function."""
        # Create a new event loop and set it
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Run the async function and get the result
            rewards = loop.run_until_complete(self.run_async(scripts, languages))
        finally:
            loop.close()

        return rewards

    async def run_async(self, scripts: List[str], languages: List[str]) -> List[float]:
        from e2b_code_interpreter import AsyncSandbox

        # Create the sandbox by hand, currently there's no context manager for this version
        try:
            sbx = await AsyncSandbox.create(timeout=30, request_timeout=3)
        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            return [0.0] * len(scripts)
        # Create a list of tasks for running scripts concurrently
        tasks = [self.run_script(sbx, script, language) for script, language in zip(scripts, languages)]

        # Wait for all tasks to complete and gather their results as they finish
        results = await asyncio.gather(*tasks)
        rewards = list(results)  # collect results

        # Kill the sandbox after all the tasks are complete
        await sbx.kill()

        return rewards

    async def run_script(self, sbx, script: str, language: str) -> float:
        try:
            execution = await sbx.run_code(script, language=language, timeout=30)
        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            return 0.0
        try:
            return float(execution.text)
        except (TypeError, ValueError):
            return 0.0

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that evaluates code snippets using the E2B code interpreter.

        Assumes the dataset contains a `verification_info` column with test cases.
        """
        evaluation_script_template = """
        import subprocess
        import json

        def evaluate_code(code, test_cases):
            passed = 0
            total = len(test_cases)
            exec_timeout = 5

            for case in test_cases:
                process = subprocess.run(
                    ["python3", "-c", code],
                    input=case["input"],
                    text=True,
                    capture_output=True,
                    timeout=exec_timeout
                )

                if process.returncode != 0:  # Error in execution
                    continue

                output = process.stdout.strip()
                if output.strip() == case["output"].strip():
                    passed += 1

            success_rate = (passed / total)
            return success_rate

        code_snippet = {code}
        test_cases = json.loads({test_cases})

        evaluate_code(code_snippet, test_cases)
        """
        verification_info = kwargs['verification_info']
        languages = [info['language'] for info in verification_info]
        code_snippets = [
            self.extract_code(completion, language) for completion, language in zip(completions, languages)
        ]
        scripts = [
            evaluation_script_template.format(
                code=json.dumps(code), test_cases=json.dumps(json.dumps(info['test_cases'])))
            for code, info in zip(code_snippets, verification_info)
        ]
        try:
            rewards = self.run_async_from_sync(scripts, languages)

        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            rewards = [0.0] * len(completions)

        return rewards


orms['external_code_reward'] = CodeReward


class CodeFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        verification_info = kwargs['verification_info']
        rewards = []
        for content, info in zip(completions, verification_info):
            pattern = r'^<think>.*?</think>\s*<answer>.*?```{}.*?```.*?</answer>(?![\s\S])'.format(info['language'])
            match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
            reward = 1.0 if match else 0.0
            rewards.append(reward)
        return rewards


orms['external_code_format'] = CodeFormat


class CodeRewardByJudge0(ORM):
    LANGUAGE_ID_MAP = {
        'assembly': 45,
        'bash': 46,
        'basic': 47,
        'c': 50,
        'c++': 54,
        'clojure': 86,
        'c#': 51,
        'cobol': 77,
        'common lisp': 55,
        'd': 56,
        'elixir': 57,
        'erlang': 58,
        'executable': 44,
        'f#': 87,
        'fortran': 59,
        'go': 60,
        'groovy': 88,
        'haskell': 61,
        'java': 62,
        'javascript': 63,
        'kotlin': 78,
        'lua': 64,
        'multi-file program': 89,
        'objective-c': 79,
        'ocaml': 65,
        'octave': 66,
        'pascal': 67,
        'perl': 85,
        'php': 68,
        'plain text': 43,
        'prolog': 69,
        'python': 71,
        'python2': 70,
        'python3': 71,
        'r': 80,
        'ruby': 72,
        'rust': 73,
        'scala': 81,
        'sql': 82,
        'swift': 83,
        'typescript': 74,
        'visual basic.net': 84
    }
    PYTHON_ID = 71

    def __init__(self):
        self.endpoint = os.getenv('JUDGE0_ENDPOINT')
        assert self.endpoint is not None, (
            'Judge0 endpoint is not set. Please set the JUDGE0_ENDPOINT environment variable.')
        x_auth_token = os.getenv('JUDGE0_X_AUTH_TOKEN')
        self.headers = {'Content-Type': 'application/json'}
        if x_auth_token is not None:
            self.headers['X-Auth-Token'] = x_auth_token

    @staticmethod
    def extract_code(completion: str, language: str) -> str:
        pattern = re.compile(rf'```{language}\n(.*?)```', re.DOTALL)
        matches = pattern.findall(completion)
        extracted_answer = matches[-1] if len(matches) >= 1 else ''
        return extracted_answer

    @classmethod
    def get_language_id(cls, language):
        if language is None:
            return cls.PYTHON_ID
        return cls.LANGUAGE_ID_MAP.get(language.lower().strip(), cls.PYTHON_ID)

    async def _evaluate_code(self, code, test_cases, language_id):
        import aiohttp
        try:
            passed = 0
            total = len(test_cases)

            for case in test_cases:
                if code is not None and code != '':
                    async with aiohttp.ClientSession() as session:
                        payload = {
                            'source_code': code,
                            'language_id': language_id,
                            'stdin': case['input'],
                            'expected_output': case['output']
                        }
                        logger.debug(f'Payload: {payload}')
                        async with session.post(
                                self.endpoint + '/submissions/?wait=true', json=payload,
                                headers=self.headers) as response:
                            response_json = await response.json()
                            logger.debug(f'Response: {response_json}')
                            if response_json['status']['description'] == 'Accepted':
                                passed += 1

            success_rate = (passed / total)
            return success_rate
        except Exception as e:
            logger.warning(f'Error from Judge0 executor: {e}')
            return 0.0

    def run_async_from_sync(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            rewards = loop.run_until_complete(self.run_async())
        finally:
            loop.close()
        return rewards

    async def run_async(self):
        tasks = [
            self._evaluate_code(code, info['test_cases'], CodeRewardByJudge0.get_language_id(info['language']))
            for code, info in zip(self.code_snippets, self.verification_info)
        ]
        results = await asyncio.gather(*tasks)
        rewards = list(results)
        return rewards

    def __call__(self, completions, **kwargs) -> List[float]:
        self.verification_info = kwargs['verification_info']

        languages = [info['language'] for info in self.verification_info]
        self.code_snippets = [
            self.extract_code(completion, language) for completion, language in zip(completions, languages)
        ]

        try:
            rewards = self.run_async_from_sync()
        except Exception as e:
            logger.warning(f'Error from Judge0 executor: {e}')
            rewards = [0.0] * len(completions)
        return rewards


orms['external_code_reward_by_judge0'] = CodeRewardByJudge0


class AsyncGenRMReward(AsyncORM):
    """
    An async reward function example that calls a generative reward model
    deployed via `swift deploy`.

    This demonstrates how to use AsyncORM with aiohttp to make parallel API calls
    to an LLM-based reward model for scoring completions.

    The reward model is prompted to evaluate each completion and output a score
    in a specific format (e.g., [[score]]).

    Usage:
        1. Deploy a reward model using swift deploy:
           ```bash
           swift deploy --model Qwen/Qwen2.5-7B-Instruct --port 8000 --infer_backend vllm
           ```

        2. Set environment variable:
           ```bash
           export GENRM_API_BASE=http://localhost:8000/v1
           ```

        3. Use in training:
           ```bash
           swift rlhf \
               --rlhf_type grpo \
               --external_plugins plugin.py \
               --reward_funcs async_genrm ...
           ```
    """

    def __init__(self):
        from openai import OpenAI
        self.api_base = os.getenv('GENRM_API_BASE', 'http://localhost:8000/v1')
        self.temperature = float(os.getenv('GENRM_TEMPERATURE', '0.3'))

        # Initialize OpenAI client to get the model name (following deepeyes_plugin pattern)
        try:
            self.client = OpenAI(
                api_key='EMPTY',
                base_url=self.api_base,
            )
            self.model_name = self.client.models.list().data[0].id
            logger.info(f'AsyncGenRMReward initialized with model: {self.model_name}')
        except Exception as e:
            raise RuntimeError('Failed to connect to the model service. Please deploy the model '
                               "using 'swift deploy --model <model_name> --port 8000 --infer_backend vllm'.") from e

        # System prompt for the generative reward model
        self.system_prompt = textwrap.dedent("""
            You are an expert evaluator. Your task is to evaluate the quality of an AI assistant's response.

            Please evaluate the response based on the following criteria:
            1. Correctness: Is the answer factually correct and logically sound?
            2. Helpfulness: Does the response address the user's question effectively?
            3. Clarity: Is the response well-organized and easy to understand?

            After your evaluation, provide a score from 0 to 10, where:
            - 0-3: Poor quality (incorrect, unhelpful, or confusing)
            - 4-6: Acceptable quality (partially correct or helpful)
            - 7-9: Good quality (correct, helpful, and clear)
            - 10: Excellent quality (perfect response)

            You MUST end your response with the score in this exact format: [[score]]
            For example: [[7]] or [[10]]
        """).strip()

    def _build_eval_prompt(self, question: str, completion: str) -> str:
        """Build the evaluation prompt for the reward model."""
        return textwrap.dedent(f"""
            ## User Question
            {question}

            ## AI Assistant's Response
            {completion}

            ## Your Evaluation
            Please evaluate the above response and provide your score.
        """).strip()

    def _extract_score(self, response: str) -> float:
        """Extract the score from the reward model's response."""
        # Look for [[score]] pattern
        match = re.search(r'\[\[(\d+(?:\.\d+)?)\]\]', response)
        if match:
            score = float(match.group(1))
            # Normalize to [0, 1] range
            return min(max(score / 10.0, 0.0), 1.0)

        # Fallback: try to find any number at the end
        match = re.search(r'(\d+(?:\.\d+)?)\s*$', response.strip())
        if match:
            score = float(match.group(1))
            return min(max(score / 10.0, 0.0), 1.0)

        logger.warning(f'Could not extract score from response: {response[:100]}...')
        return 0.0

    async def _score_single(self, session, question: str, completion: str) -> float:
        """Score a single completion using the generative reward model."""
        import aiohttp

        eval_prompt = self._build_eval_prompt(question, completion)

        payload = {
            'model': self.model_name,
            'messages': [{
                'role': 'system',
                'content': self.system_prompt
            }, {
                'role': 'user',
                'content': eval_prompt
            }],
            'temperature': self.temperature,
            'max_tokens': 2048,
            'seed': random.randint(0, 1000000),
        }

        try:
            async with session.post(
                    f'{self.api_base}/chat/completions', json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.warning(f'API error {resp.status}: {error_text[:200]}')
                    return 0.0

                result = await resp.json()
                response_content = result['choices'][0]['message']['content']
                return self._extract_score(response_content)

        except asyncio.TimeoutError:
            logger.warning('API request timed out')
            return 0.0
        except Exception as e:
            logger.warning(f'Error calling reward model API: {e}')
            return 0.0

    async def __call__(self, completions, messages, **kwargs) -> List[float]:
        """
        Score completions using a generative reward model via async API calls.

        Args:
            completions: List of model-generated responses
            messages: List of conversation messages (used to extract the question)
            **kwargs: Additional arguments (unused)

        Returns:
            List of reward scores in [0, 1] range
        """
        import aiohttp

        # Extract questions from messages (assuming the last user message is the question)
        questions = []
        for msg_list in messages:
            question = ''
            for msg in reversed(msg_list):
                if msg.get('role') == 'user':
                    question = msg.get('content', '')
                    break
            questions.append(question)

        # Make parallel API calls using asyncio.gather
        async with aiohttp.ClientSession() as session:
            tasks = [self._score_single(session, q, c) for q, c in zip(questions, completions)]
            rewards = await asyncio.gather(*tasks)
            return list(rewards)


orms['async_genrm'] = AsyncGenRMReward


# ref implementation: https://github.com/qiancheng0/ToolRL/blob/main/verl/utils/reward_score/rlla.py
# arxiv paper: https://arxiv.org/abs/2504.13958
# MAX1STEP30MAX3: enable Two stage reward Setting include Format and Correctness
# SCHEDULEREWARD: enable Dynamic (Finegrained) reward Setting include Format and Correctness
# Correctness Reward Granularity:
# COARSEREWARD -> Coarse, INTERMEDIATEREWARD -> Intermediate, REFINEDREWARD -> Finegrained
class ToolUseFormatReward(ORM):

    def __init__(self):
        self.format_max_possible = 1.0
        self.format_min_possible = 0.0

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        trainer_state = kwargs.get('trainer_state')
        global_step = trainer_state.global_step
        max_possible_reward = self.format_max_possible
        min_possible_reward = self.format_min_possible
        # Two stage (Coarse) Setting, divide training into two phases. Format Reward in [0,0.5] if step < 30 else [0,1]
        if str(os.getenv('MAX1STEP30MAX3', 0)) == '1':
            if global_step >= 30:
                max_possible_reward = self.format_max_possible / 2
                min_possible_reward = self.format_min_possible / 2
            else:
                max_possible_reward = self.format_max_possible
                min_possible_reward = self.format_min_possible

        # apply continuous interpolation between the two reward scales throughout training.
        if str(os.getenv('SCHEDULEREWARD', 0)) == '1':
            max_possible_reward = 2 - (2 - max_possible_reward) * global_step / 150
            min_possible_reward = -2 + (2 + min_possible_reward) * global_step / 150
            if max_possible_reward < 1.0:
                max_possible_reward = 1.0
            if min_possible_reward > -1.0:
                min_possible_reward = -1.0

        rewards = []
        responses = completions

        for response, ans in zip(responses, solution):
            reward = min_possible_reward
            if '<response>' in ans and '<tool_call>' not in ans:
                pattern = r'^<think>.*?</think>\s*<response>.*?</response>$'
                if re.search(pattern, response,
                             re.DOTALL) and response.count('<response>') == 1 and response.count('</response>') == 1:
                    reward = max_possible_reward
            elif '<response>' not in ans and '<tool_call>' in ans:
                pattern = r'^<think>.*?</think>\s*<tool_call>.*?</tool_call>$'
                if re.search(pattern, response,
                             re.DOTALL) and response.count('<tool_call>') == 1 and response.count('</tool_call>') == 1:
                    reward = max_possible_reward
            elif '<response>' in ans and '<tool_call>' in ans:
                pattern = r'^<think>.*?</think>\s*<tool_call>.*?</tool_call>\s*<response>.*?</response>$'
                if (re.search(pattern, response, re.DOTALL) and response.count('<tool_call>') == 1
                        and response.count('</tool_call>') == 1 and response.count('<response>') == 1
                        and response.count('</response>') == 1):
                    reward = max_possible_reward
            else:
                pattern = r'^<think>.*?</think>$'
                if re.search(pattern, response, re.DOTALL):
                    reward = max_possible_reward

            rewards.append(reward)

        return rewards


orms['external_tooluse_format_reward'] = ToolUseFormatReward


class ToolUseLengthReward(ORM):

    def __init__(self):
        self.length_max_possible = 1.0
        self.length_min_possible = 0.0

    # customized reward functions: length
    def __call__(self, completions, solution, **kwargs):
        max_possible_reward = self.length_max_possible
        min_possible_reward = self.length_min_possible
        trainer_state = kwargs.get('trainer_state')
        global_step = trainer_state.global_step
        # SCHEDULELENGTH: enable Dynamic Length Reward
        if os.getenv('SCHEDULELENGTH', 0) == '1':
            max_reward_len = (640 - 384) * global_step / 105 + 384
        else:
            max_reward_len = 512
        """Reward function that gives higher scores to longer completions."""
        responses = completions
        rewards = []

        for response, ans in zip(responses, solution):
            if '<think>' not in response or '</think>' not in response:
                rewards.append(min_possible_reward)
                continue
            think_responses = response.split('<think>')[-1].split('</think>')[0].strip()
            reward = round(len(think_responses.split()) / max_reward_len, 2)
            if reward > 1.0:
                reward = 1.0

            final_reward = reward * (max_possible_reward - min_possible_reward) + min_possible_reward
            rewards.append(final_reward)

        return rewards


orms['external_tooluse_length_reward'] = ToolUseLengthReward


class ToolUseCorrectnessReward(ORM):

    def __init__(self):
        if str(os.getenv('CORRECTMAX1', 0)) == '1':
            self.tool_max_possible = 1.0
            self.tool_min_possible = -1.0
        else:
            self.tool_max_possible = 3.0
            self.tool_min_possible = -3.0

    def match_score(self, list1, list2):
        if list1 == list2:
            return 1.0

        if os.getenv('REFINEDREWARD', 0) == '1':
            if list1 != list2:
                return 0.0

        if not list1 or not list2:
            return 0.0

        count1 = Counter(list1)  # Frequency count for list1
        count2 = Counter(list2)  # Frequency count for list2

        intersection = sum(min(count1[k], count2[k]) for k in count1.keys() & count2.keys())
        max_possible = len(list1) + len(list2) - intersection

        return intersection / max_possible if max_possible > 0 else 0.0

    def compute_tool_call_reward(self, gt_tools, pd_tools, max_possible_reward, min_possible_reward):
        if gt_tools == pd_tools:
            return max_possible_reward

        if os.getenv('COARSEREWARD', 0) == '1':
            if gt_tools != pd_tools:
                return min_possible_reward

        gt_names = [tool['name'] for tool in gt_tools]
        pd_names = [tool['name'] for tool in pd_tools]
        score = self.match_score(list(gt_names), list(pd_names))

        local_max_possible = 1.0
        used_pd_indices = set()  # Keep track of matched pd_tools

        for gt_tool in gt_tools:
            gt_name = gt_tool['name']
            gt_params = gt_tool['parameters']

            if str(os.getenv('INTERMEDIATEREWARD', 0)) == '1':
                local_max_possible += 1.0
            else:
                local_max_possible += 1.0 + len(gt_params)

            best_match = None
            best_match_score = 0.0
            best_match_index = -1

            # Find the best matching unused pd_tool
            for i, pd_tool in enumerate(pd_tools):
                if i in used_pd_indices or pd_tool['name'] != gt_name:
                    continue

                if str(os.getenv('INTERMEDIATEREWARD', 0)) == '1':
                    if gt_tool == pd_tool:
                        best_match = pd_tool
                        best_match_index = i
                        best_match_score = 1.0
                        break
                    else:
                        continue

                pd_params = pd_tool['parameters']
                param_score = self.match_score(list(gt_params.keys()), list(pd_params.keys()))

                # Calculate correctness score for parameter values
                correctness_score = sum(1.0 for k, v in gt_params.items() if k in pd_params and pd_params[k] == v)

                total_score = param_score + correctness_score

                if total_score > best_match_score:
                    best_match_score = total_score
                    best_match = pd_tool
                    best_match_index = i

            if best_match:
                used_pd_indices.add(best_match_index)
                score += best_match_score

        return (max_possible_reward - min_possible_reward) * score / local_max_possible + min_possible_reward

    # custoimzed reward functions: tool call correctness
    def __call__(self, completions, solution, **kwargs):
        trainer_state = kwargs.get('trainer_state')
        global_step = trainer_state.global_step
        max_possible_reward = self.tool_max_possible
        min_possible_reward = self.tool_min_possible
        # two stage (Coarse) Setting, divide training into two phases.
        if str(os.getenv('MAX1STEP30MAX3', 0)) == '1':
            if global_step < 30:
                max_possible_reward = max_possible_reward / 3
                min_possible_reward = min_possible_reward / 3
            else:
                max_possible_reward = max_possible_reward
                min_possible_reward = min_possible_reward
        # apply continuous interpolation between the two reward scales throughout training.
        if str(os.getenv('SCHEDULEREWARD', 0)) == '1':
            max_possible_reward = (max_possible_reward - 2) * global_step / 150 + 2
            min_possible_reward = (min_possible_reward + 2) * global_step / 150 - 2
            if max_possible_reward > 3.0:
                max_possible_reward = 3.0
            if min_possible_reward < -3.0:
                min_possible_reward = -3.0

        responses = completions
        rewards = []

        for response, ans in zip(responses, solution):
            reward = 0.0

            if '<tool_call>' not in ans:
                # if "<tool_call>" not in response and "</tool_call>" not in response:
                #     reward = max_possible_reward
                # else:
                #     reward = min_possible_reward
                rewards.append(reward)
                continue

            gt_tool_call = ans.split('<tool_call>')[1].split('</tool_call>')[0].strip()
            gt_tools = gt_tool_call.split('\n')
            gt_tools = [json.loads(tool) for tool in gt_tools]  # each diction contains "name" and "parameter"

            try:
                # if the format is not correct, directly give the lowest possible score
                assert '<tool_call>' in response
                assert '</tool_call>' in response
                pd_tools = response.split('<tool_call>')[1].split('</tool_call>')[0].strip().split('\n')
                pd_tools = [json.loads(tool) for tool in pd_tools]
                reward = self.compute_tool_call_reward(gt_tools, pd_tools, max_possible_reward,
                                                       min_possible_reward)  # top reward is 2
            except (ValueError, IndexError, AssertionError):
                reward = min_possible_reward

            rewards.append(reward)

        return rewards


orms['external_tooluse_correct_reward'] = ToolUseCorrectnessReward
"""
TO CUSTOMIZE REWARD MODEL:
    Step 1: Define a Reward Class
        Implement your custom reward calculation logic within the __call__ method.
        The method accepts the messages generated by the model during interactions
        and dataset columns as inputs parameters.

    Step 2: Add your reward model plugin to the rm_plugins registry:
        rm_plugins['my_rm_plugin'] = MyRMPlugin

    Step 3: Configure the Arguments
        Run the script with:
        --external_plugins "Path to your plugin" \
        --reward_model_plugin my_rm_plugin

For GenRM you can refer to swift/llm/plugin/rm_plugin/GenRMPlugin
"""


class CustomizedRMPlugin:
    """
    Customized Reward Model Plugin, same to DefaultRMPlugin

    It assumes that `self.model` is a classification model with a value head(output dimmension 1).
    The first logits value from the model's output is used as the reward score.
    """

    def __init__(self, model, template):
        self.model = model
        self.template: Template = template

    def __call__(self, inputs, **kwargs):
        batched_inputs = [self.template.encode(deepcopy(infer_request)) for infer_request in inputs]
        reward_inputs = to_device(self.template.data_collator(batched_inputs), self.model.device)

        with torch.inference_mode():
            return self.model(**reward_inputs).logits[:, 0]


class QwenLongPlugin(DefaultRMPlugin):
    # https://arxiv.org/abs/2505.17667
    # NOTE: you should customize the verified reward function, you can refer to
    # https://github.com/Tongyi-Zhiwen/QwenLong-L1/tree/main/verl/verl/utils/reward_score
    # hf_dataset: https://huggingface.co/datasets/Tongyi-Zhiwen/DocQA-RL-1.6K/viewer/default/train
    # ms_dataset: https://modelscope.cn/datasets/iic/DocQA-RL-1.6K
    def __init__(self, model, template, accuracy_orm=None):
        super().__init__(model, template)
        # initilize PTEngine to infer
        self.engine = PtEngine.from_model_template(self.model, self.template, max_batch_size=0)  # 0: no limit
        self.request_config = RequestConfig(temperature=0)  # customise your request config here
        self.system = textwrap.dedent("""
            You are an expert in verifying if two answers are the same.

            Your input consists of a problem and two answers: Answer 1 and Answer 2.
            You need to check if they are equivalent.

            Your task is to determine if the two answers are equivalent, without attempting to solve the original problem.
            Compare the answers to verify they represent identical values or meanings,
            even when expressed in different forms or notations.

            Your output must follow this format:
            1) Provide an explanation for why the answers are equivalent or not.
            2) Then provide your final answer in the form of: [[YES]] or [[NO]]

            Problem: {problem_placeholder}
            Answer 1: {answer1_placeholder}
            Answer 2: {answer2_placeholder}
        """)  # noqa
        self.accuracy_orm = accuracy_orm

    def __call__(self, inputs, **kwargs):
        completions = [example['messages'][-1]['content'] for example in inputs]
        ground_truths = [example['reward_model']['ground_truth'] for example in inputs]
        rm_inputs = self.prepare_rm_inputs(inputs, completions, ground_truths)

        results = self.engine.infer(rm_inputs, self.request_config, use_tqdm=False)
        llm_rewards = self.compute_rewards(results)

        if self.accuracy_orm:
            verified_rewards = self.accuracy_orm(completions, ground_truths)
        else:
            verified_rewards = [0.0] * len(llm_rewards)

        rewards = [max(r1, r2) for r1, r2 in zip(llm_rewards, verified_rewards)]
        return torch.tensor(rewards, dtype=torch.float32)

    def prepare_rm_inputs(self, inputs: List[Dict], completions, ground_truths) -> List[Dict]:
        rm_inputs = []
        for infer_request, completion, ground_truth in zip(inputs, completions, ground_truths):
            # Deep copy to prevent modification of original input
            rm_infer_request = deepcopy(infer_request)
            problem = infer_request['messages'][0]['content']
            start_index = problem.index('</text>')
            end_index = problem.index('Format your response as follows:')
            question = problem[start_index:end_index].replace('</text>', '').strip()
            prompt = self.system.format(
                problem_placeholder=question, answer1_placeholder=completion, answer2_placeholder=ground_truth)

            # Construct new messages tailored for the reward model
            rm_messages = [{'role': 'user', 'content': prompt}]

            # Update the messages in the reward infer request
            rm_infer_request['messages'] = rm_messages
            rm_inputs.append(rm_infer_request)
        return rm_inputs

    @staticmethod
    def extract_reward(model_output: str) -> float:
        match = re.search(r'\[([A-Z]+)\]', model_output)
        if match:
            answer = match.group(1)
            if answer == 'YES':
                return 1.0
            elif answer == 'NO':
                return 0.0
            else:
                logger.warning("Unexpected answer, expected 'YES' or 'NO'.")
                return 0.0
        else:
            logger.warning("Unable to extract reward score from the model's output, setting reward to 0")
            return 0.0  # Or raise ValueError("Format incorrect")

    def compute_rewards(self, results: List[ChatCompletionResponse]) -> List[float]:
        """
        Compute average reward scores from the reward model's outputs.

        Args:
            results (List[ChatCompletionResponse]): A list of results from the reward model.

        Returns:
            List[float]: A list of average reward scores.
        """
        rewards = []
        for idx, output in enumerate(results):
            try:
                cur_rewards = []
                for choice in output.choices:
                    response = choice.message.content
                    reward = self.extract_reward(response)
                    cur_rewards.append(reward)
                cur_rewards = [r for r in cur_rewards if r is not None]
                if cur_rewards:
                    average_reward = sum(cur_rewards) / len(cur_rewards)
                else:
                    average_reward = 0.0
                    logger.warning('No valid rewards extracted. Assigning reward score of 0.0.')

                rewards.append(average_reward)
            except Exception as e:
                logger.error(f'Error computing reward: {e}')
                rewards.append(0.0)  # Assign default reward score on failure
        return rewards


rm_plugins['my_rmplugin'] = CustomizedRMPlugin
rm_plugins['qwenlong'] = QwenLongPlugin
"""
TO CUSTOMIZE MULTITURN SCHEDULER:
    Step 1: Define a Scheduler Class
        Implement your custom scheduler with the following methods:
            - step (Required): Constructs the next round of the infer request.
            - check_finished (Optional): Determines whether the current round has finished,
                which defaults to ending when the inference result is truncated (over length) or
                when the maximum number of rounds is reached.
            or override run method in MultiTurnScheduler class.

        Both methods accept:
            - the last turn's InferRequest/response_choice
            - the current turn count

    Step 2: Add your scheduler to the multi_turns registry:
        multi_turns['my_scheduler'] = MyScheduler

    Step 3: Configure the Arguments
        Run the script with:
        swift rollout \
            --external_plugins "Path to your plugin" \
            --multi_turn_scheduler my_scheduler
"""


class ToolCallScheduler(MultiTurnScheduler):
    # A simple scheduler that supports tool calls by overriding the `step` method
    # Tool parsing uses the ReAct format
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # A simple tool registry. Extend or replace with your own tools as needed.
        self.tools = {
            'calculator': self._calculator_tool,
        }

    def _calculator_tool(self, expression: str) -> str:
        # A very small sandboxed calculator
        # The calculator tool implemented here can perform only basic arithmetic operations and
        # may not be able to solve all math problems in the dataset.
        import ast
        import operator

        def _evaluate_ast_node(node) -> Union[int, float]:
            operators = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.USub: operator.neg,
                ast.UAdd: operator.pos,
            }

            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return node.value
                else:
                    raise TypeError(f'Unsupported constant type: {type(node.value)}')

            elif isinstance(node, ast.Num):
                return node.n

            elif isinstance(node, ast.BinOp):
                left = _evaluate_ast_node(node.left)
                right = _evaluate_ast_node(node.right)
                op = operators.get(type(node.op))

                if op is None:
                    raise TypeError(f'Unsupported operation: {type(node.op).__name__}')

                if isinstance(node.op, ast.Div) and right == 0:
                    raise ZeroDivisionError('Division by zero')

                return op(left, right)

            elif isinstance(node, ast.UnaryOp):
                operand = _evaluate_ast_node(node.operand)
                op = operators.get(type(node.op))

                if op is None:
                    raise TypeError(f'Unsupported unary operation: {type(node.op).__name__}')

                return op(operand)

            else:
                raise TypeError(f'Unsupported AST node type: {type(node).__name__}')

        try:
            expression = expression.strip().replace(' ', '')

            if not re.match(r'^[0-9+\-*/().\s]+$', expression):
                return 'Error: expression contains disallowed characters.'

            if expression.count('(') != expression.count(')'):
                return 'Error: unmatched parentheses.'

            try:
                result = ast.literal_eval(expression)
                return f'Result: {result}'
            except (ValueError, SyntaxError):
                node = ast.parse(expression, mode='eval')
                result = _evaluate_ast_node(node.body)
                return f'Result: {result}'

        except Exception as e:
            return f'Calculation error: {e}'

    def _extract_tool_calls(self, text: str):
        """
        Parse tool-call patterns using ReAct format from model output.
        Format: Action: tool_name\nAction Input: parameters
        """
        import re

        pattern = r'Action:\s*(.*?)\s*\nAction Input:\s*(.*?)(?:\n|$)'
        matches = re.findall(pattern, text, re.DOTALL)
        if not matches:
            return None
        return [{'tool': name.strip(), 'params': params.strip()} for name, params in matches]

    def _execute_tools(self, tool_calls):
        """Run each requested tool and collect its observation string."""
        results = []
        for call in tool_calls:
            name, params = call['tool'], call['params']
            if name in self.tools:
                try:
                    result = self.tools[name](params)
                    results.append(result)
                except Exception as e:
                    results.append(f'tool error {e}')
            else:
                results.append(f'unknown tool {name}')
        return results

    def check_finished(self, infer_request: 'RolloutInferRequest', response_choice: 'ChatCompletionResponseChoice',
                       current_turn: int) -> bool:
        completion = response_choice.message.content
        tool_calls = self._extract_tool_calls(completion)
        if tool_calls is None:
            return True

        return super().check_finished(infer_request, response_choice, current_turn)

    def step(self, infer_request: 'RolloutInferRequest', response_choice: 'ChatCompletionResponseChoice',
             current_turn: int) -> Dict:
        completion = response_choice.message.content
        token_ids = response_choice.token_ids
        loss_mask = [1] * len(token_ids)
        tool_calls = self._extract_tool_calls(completion)
        # assert len(tool_calls) == 1, 'this scheduler is designed for one tool call per turn'
        tool_results = self._execute_tools(tool_calls)
        # append tool result to the completion
        infer_request.messages[-1]['content'] += (tool_results[0])

        tokenizer = self.tokenizer
        result_tokens = tokenizer.encode(tool_results[0], add_special_tokens=False)
        token_ids.extend(result_tokens)
        loss_mask.extend([0] * len(result_tokens))

        return {
            'infer_request': infer_request,
            'response_token_ids': token_ids,
            'response_loss_mask': loss_mask,
            'rollout_infos': {
                'tool_results': tool_results[0],
                'num_turns': current_turn,
            }
        }


multi_turns['tool_call_scheduler'] = ToolCallScheduler


# register GYM env
class CustomEnv(Env):
    pass


envs['custom_env'] = CustomEnv


class CustomCtxManager(ContextManager):
    pass


context_managers['custom_ctx'] = CustomCtxManager
