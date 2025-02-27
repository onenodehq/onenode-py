class VisionModels:
    """
    VisionModels - Constants for supported vision models in CapybaraDB
    
    This class provides constants for the vision models supported by CapybaraDB.
    These models are used to process and understand image content for multimodal applications.
    
    When creating an EmbImage object, you can specify which vision model to use.
    Different models have different capabilities in terms of:
    - Image understanding quality
    - Processing speed
    - Cost
    - Feature support
    
    Usage:
        ```python
        from capybaradb import EmbImage, VisionModels
        
        # Create an EmbImage with a specific vision model
        image = EmbImage(
            "base64_encoded_image_data",
            vision_model=VisionModels.GPT_4O
        )
        ```
    """
    
    # OpenAI's GPT-4o model
    # A multimodal model that can process both text and images.
    # Provides high-quality image understanding and can generate
    # detailed descriptions and insights from visual content.
    GPT_4O = "gpt-4o"
    
    # OpenAI's GPT-4o-mini model
    # A smaller, faster version of GPT-4o with reduced capabilities.
    # Good for applications where speed is more important than
    # the most detailed image understanding.
    GPT_4O_MINI = "gpt-4o-mini"
    
    # OpenAI's GPT-4-turbo model
    # An optimized version of GPT-4 with vision capabilities.
    # Balances performance and quality for most vision applications.
    GPT_4O_TURBO = "gpt-4-turbo"
    
    # OpenAI's o1 model
    # The most advanced vision model with enhanced capabilities
    # for complex visual reasoning and understanding.
    GPT_O1 = "o1"
