from dotenv import load_dotenv
import os
from langfuse import Langfuse

class PromptVerificationError(Exception):
    """Custom exception for prompt verification failures"""
    pass

def main():
    # Load environment variables
    load_dotenv()

    # Initialize clients
    hipaa_client = Langfuse(
        secret_key=os.getenv('HIPAA_SECRET_KEY'),
        public_key=os.getenv('HIPAA_PUBLIC_KEY'),
        host=os.getenv('HIPAA_HOST')
    )

    nonpii_client = Langfuse(
        secret_key=os.getenv('NONPII_SECRET_KEY'),
        public_key=os.getenv('NONPII_PUBLIC_KEY'),
        host=os.getenv('NONPII_HOST')
    )

    # Get all prompts from HIPAA environment
    hipaa_prompts = hipaa_client.client.prompts.list()
    
    # Filter for production prompts
    production_prompts = [p for p in hipaa_prompts.data if "production" in p.labels]
    
    if not production_prompts:
        print("No production prompts found in HIPAA environment")
        return

    print(f"\nFound {len(production_prompts)} production prompts in HIPAA environment")
    
    # Track any verification failures
    failures = []
    verified = []

    # Verify each production prompt
    for prompt in production_prompts:
        try:
            # Get full prompt details from HIPAA
            hipaa_details = hipaa_client.get_prompt(prompt.name, label="production")
            
            # Try to get the prompt from NON-PII
            try:
                nonpii_details = nonpii_client.get_prompt(prompt.name, label="production")
                
                # Compare prompt and config
                if hipaa_details.prompt != nonpii_details.prompt:
                    error_msg = f"❌ Prompt '{prompt.name}' has different prompt content"
                    print(error_msg)
                    failures.append(error_msg)
                    continue
                    
                if hipaa_details.config != nonpii_details.config:
                    error_msg = f"❌ Prompt '{prompt.name}' has different config"
                    print(error_msg)
                    failures.append(error_msg)
                    continue
                
                # Verify both have production label
                if "production" not in hipaa_details.labels or "production" not in nonpii_details.labels:
                    error_msg = f"❌ Prompt '{prompt.name}' missing production label in one environment"
                    print(error_msg)
                    failures.append(error_msg)
                    continue
                
                success_msg = f"✅ Prompt '{prompt.name}' verified successfully"
                print(success_msg)
                verified.append(success_msg)
                
            except Exception as e:
                error_msg = f"❌ Prompt '{prompt.name}' not found in NON-PII environment: {str(e)}"
                print(error_msg)
                failures.append(error_msg)
                
        except Exception as e:
            error_msg = f"❌ Failed to verify prompt '{prompt.name}': {str(e)}"
            print(error_msg)
            failures.append(error_msg)

    # Print summary
    print("\nVerification Summary:")
    print(f"Total prompts processed: {len(production_prompts)}")
    print(f"Successfully verified: {len(verified)}")
    print(f"Failed to verify: {len(failures)}")

    # If there were any failures, raise an exception
    if failures:
        raise PromptVerificationError("\n".join(failures))

if __name__ == "__main__":
    try:
        main()
        print("\n✅ All prompts verified successfully!")
    except PromptVerificationError as e:
        print("\n❌ Verification failed with the following errors:")
        print(str(e))
        exit(1)