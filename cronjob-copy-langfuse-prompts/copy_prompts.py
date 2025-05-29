from dotenv import load_dotenv
import os
import pdb
from langfuse import Langfuse

class PromptCopyError(Exception):
    """Custom exception for prompt copy failures"""
    pass

def are_prompts_different(hipaa_details, nonpii_details):
    """Compare prompt content and config between environments"""
    if hipaa_details.prompt != nonpii_details.prompt:
        return True
    if hipaa_details.config != nonpii_details.config:
        return True
    return False

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

    # Get all prompts from HIPAA environment with pagination
    all_prompts = []
    page = 1
    
    while True:
        hipaa_prompts = hipaa_client.client.prompts.list(page=page, limit=50)
        all_prompts.extend(hipaa_prompts.data)
        
        # Check if we've reached the last page
        if page >= hipaa_prompts.meta.total_pages:
            break
        page += 1
    
    print(f"Retrieved {len(all_prompts)} total prompts from HIPAA environment")
    
    # Filter for production prompts
    production_prompts = [p for p in all_prompts if "production" in p.labels]
    
    if not production_prompts:
        print("No production prompts found in HIPAA environment")
        return

    print(f"\nFound {len(production_prompts)} production prompts in HIPAA environment")
    
    # Track any copy failures and stats
    failures = []
    skipped = []
    copied = []

    # Process each production prompt
    for prompt in production_prompts:
        try:
            # Get full prompt details from HIPAA
            hipaa_details = hipaa_client.get_prompt(prompt.name, label="production")
            
            # Try to get the prompt from NON-PII
            try:
                nonpii_details = nonpii_client.get_prompt(prompt.name, label="production")
                
                # Skip if content is identical
                if not are_prompts_different(hipaa_details, nonpii_details):
                    skip_msg = f"⏭️  Skipping prompt '{prompt.name}' - content is identical"
                    print(skip_msg)
                    skipped.append(skip_msg)
                    continue
            except Exception:
                # Prompt doesn't exist in NON-PII, will be created
                pass
            
            # Create or update in NON-PII environment
            nonpii_client.create_prompt(
                name=prompt.name,
                prompt=hipaa_details.prompt,
                config=hipaa_details.config,
                labels=["production"]  # Only copy the production label
            )
            
            success_msg = f"✅ Successfully copied prompt '{prompt.name}'"
            print(success_msg)
            copied.append(success_msg)
            
        except Exception as e:
            error_msg = f"❌ Failed to copy prompt '{prompt.name}': {str(e)}"
            print(error_msg)
            failures.append(error_msg)

    # Print summary
    print("\nCopy Summary:")
    print(f"Total prompts processed: {len(production_prompts)}")
    print(f"Successfully copied: {len(copied)}")
    print(f"Skipped (identical): {len(skipped)}")
    print(f"Failed to copy: {len(failures)}")

    # If there were any failures, raise an exception
    if failures:
        raise PromptCopyError("\n".join(failures))

if __name__ == "__main__":
    try:
        main()
        print("\n✅ All prompts processed successfully!")
    except PromptCopyError as e:
        print("\n❌ Copy operation failed with the following errors:")
        print(str(e))
        exit(1) 