import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

class PromptManager:
    """ Manages versioned prompts stored as YAML files."""

    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = Path(prompts_dir)
        if not self.prompts_dir.exists():
            self.prompts_dir.mkdir(parents=True, exist_ok=True)
        
    def load_prompt(self, agent_name: str, version: str = "current") -> Dict[str, Any]:
        """
            Load a versioned prompt from disk.

            Args:
                agent_name: Name of the agent (e.g., "customer_support")
                version: Version to load (e.g., "v1.0.0" or "current")
            
            Returns:
                Dictionary containing prompt data
        """
        if version == "current":
            prompt_file = self.prompts_dir / agent_name / "current.yaml"
        else:
            prompt_file = self.prompts_dir / agent_name / f"{version}.yaml"
        
        if not prompt_file.exists():
            raise ValueError(f"Prompt not found: {prompt_file}")

        with open(prompts_file, 'r', encoding='utf-8') as f:
            prompt_data = yaml.safe_load(f)
        
        # Add metadata
        prompt_data['loaded_at'] = datetime.utcnow().isoformat()
        prompt_data['file_path'] = str(prompt_file)

        return prompt_data
    
    def get_version_history(self, agent_name: str) -> list[str]:
        """List all versions for an agent."""
        agent_dir = self.prompts_dir / agent_name

        if not agent_dir.exists():
            return []
        
        versions = [
            f.stem for f in agent_dir.glob("v*.yml")
        ]
        return sorted(versions, reverse = True)
    
    def compile_prompt(self, prompt_data: Dict[str,Any], user_message: str) -> str:
        """
        Compile the 4-layer prompt into final text.
        
        Layers:
        1. Security (top) - Sandwich defense
        2. Role & Constraints
        3. Context & Examples
        4. Task (current request)
        5. Security (bottom) - Final check
        """

        # Layer 4 (Top): Security Guards
        security_top = prompt_data.get('security', {}).get('top_guard', '')

        #Layer 1 : Role & Constraints
        role_section = self._format_role(prompt_data.get('role', {}))
        constraints_section = self._format_constraints(prompt_data.get('constraints', {}))

        #Layer 2: Context & Examples
        context_section = self._format_context(prompt_data.get('context', {}))
        examples_section = self._format_examples(prompt_data.get('examples', []))

        # Layer 3: Current Task
        task_section = f"""
            CURRENT REQUEST:
            User: {user_message}

            Your task: Analyze this request and respond according to the guidelines above.
        """

        # Layer 4 (Bottom): Security Guards
        security_bottom = prompt_data.get('security', {}).get('bottom_guard', '')

        full_prompt = f"""
            {security_top}

            {role_section}

            {constraints_section}

            {context_section}

            {examples_section}

            {task_section}

            {security_bottom}
        """
        
        return full_prompt.strip()
    
    def _format_role(self, role: Dict) -> str:
        """Format Layer 1: Role section."""
        if not role:
            return ""
        
        identity = role.get('identity', '')
        expertise = role.get('expertise', '')
        tone = role.get('tone', '')
        
        return f"""
ROLE:
You are: {identity}
Your expertise: {expertise}
Your tone: {tone}
"""
    
    def _format_constraints(self, constraints: Dict) -> str:
        """Format Layer 1: Constraints section."""
        if not constraints:
            return ""
        
        lines = ["CONSTRAINTS:"]
        
        if 'monetary_limit' in constraints:
            lines.append(f"- Monetary limit: {constraints['monetary_limit']}")
        
        if 'data_access' in constraints:
            lines.append(f"- Data access: {constraints['data_access']}")
        
        if 'scope' in constraints:
            lines.append(f"- Scope: {constraints['scope']}")
        
        if 'prohibited_actions' in constraints:
            lines.append("- Prohibited actions:")
            for action in constraints['prohibited_actions']:
                lines.append(f"  • {action}")
        
        return "\n".join(lines)
    
    def _format_context(self, context: Dict) -> str:
        """Format Layer 2: Context section."""
        if not context:
            return ""
        
        lines = ["CONTEXT:"]
        
        if 'company_info' in context:
            lines.append("Company Information:")
            for info in context['company_info']:
                lines.append(f"  • {info}")
        
        if 'processes' in context:
            lines.append("\nProcess Flows:")
            for process_name, steps in context['processes'].items():
                lines.append(f"  {process_name}:")
                for step in steps:
                    lines.append(f"    - {step}")
        
        return "\n".join(lines)
    
    def _format_examples(self, examples: list) -> str:
        """Format Layer 2: Examples section."""
        if not examples:
            return ""
        
        lines = ["EXAMPLES:"]
        
        for i, example in enumerate(examples, 1):
            scenario = example.get('scenario', '')
            user = example.get('user', '')
            correct_response = example.get('correct_response', {})
            
            lines.append(f"\nExample {i}: {scenario}")
            lines.append(f"  User: {user}")
            lines.append(f"  Correct Response:")
            lines.append(f"    Reasoning: {correct_response.get('reasoning', '')}")
            lines.append(f"    Action: {correct_response.get('action', '')}")
            lines.append(f"    Message: {correct_response.get('message', '')}")
        
        return "\n".join(lines)
