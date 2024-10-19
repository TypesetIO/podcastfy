"""
Podcastfy CLI

This module provides a command-line interface for generating podcasts or transcripts
from URLs or existing transcript files. It orchestrates the content extraction,
generation, and text-to-speech conversion processes.
"""

from multiprocessing import Pool
import os
import uuid
import typer
import yaml
import time
from podcastfy.content_parser.content_extractor import ContentExtractor
from podcastfy.content_generator import ContentGenerator
from podcastfy.text_to_speech import TextToSpeech
from podcastfy.utils.config import Config, load_config
from podcastfy.utils.config_conversation import (
    ConversationConfig,
    load_conversation_config,
)
from podcastfy.utils.logger import setup_logger
from typing import List, Optional, Dict, Any
import copy


logger = setup_logger(__name__)

app = typer.Typer()

# Function to split the list of files into chunks
def chunkify(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def process_content(
    paths_list=None,
    transcript_file=None,
    output_path=None,
    tts_model="openai",
    generate_audio=True,
    config=None,
    conversation_config: Optional[Dict[str, Any]] = None,
    image_paths: Optional[List[str]] = None,
    is_local: bool = False,
):
    """
    Process URLs, a transcript file, or image paths to generate a podcast or transcript.

    Args:
        paths_list (Optional[List[str]]): A list of paths to process.
        transcript_file (Optional[str]): Path to a transcript file.
        tts_model (str): The TTS model to use ('openai', 'elevenlabs' or 'edge'). Defaults to 'openai'.
        generate_audio (bool): Whether to generate audio or just a transcript. Defaults to True.
        config (Config): Configuration object to use. If None, default config will be loaded.
        conversation_config (Optional[Dict[str, Any]]): Custom conversation configuration.
        image_paths (Optional[List[str]]): List of image file paths to process.
        is_local (bool): Whether to use a local LLM. Defaults to False.

    Returns:
        Optional[str]: Path to the final podcast audio file, or None if only generating a transcript.
    """
    st = time.time()
    logger.info(f'paths list: {paths_list}')
    try:
        if config is None:
            config = load_config()
        
        # Load default conversation config
        conv_config = load_conversation_config()
        
        # Update with provided config if any
        if conversation_config:
            conv_config.configure(conversation_config)

        if transcript_file:
            logger.info(f"Using transcript file: {transcript_file}")
            with open(transcript_file, "r") as file:
                qa_content = file.read()
        else:
            content_generator = ContentGenerator(
                api_key=config.GEMINI_API_KEY, conversation_config=conv_config.to_dict()
            )

            if paths_list:
                logger.info(f"Processing {len(paths_list)} links")
                for path in paths_list:
                
                    content_extractor = ContentExtractor()
                    # Extract content from links
                    combined_content = content_extractor.extract_content(path)
                    
                    # Generate Q&A content
                    random_filename = f"transcript_{uuid.uuid4().hex}.txt"
                    transcript_filepath = os.path.join(
                        config.get("output_directories")["transcripts"], random_filename
                    )
                    qa_content = content_generator.generate_qa_content(
                        combined_content,
                        image_file_paths=image_paths or [],
                        output_filepath=transcript_filepath,
                        is_local=is_local,
                    )
                    
                    if generate_audio:
                        api_key = None
                        # edge does not require an API key
                        if tts_model != "edge":
                            api_key = getattr(config, f"{tts_model.upper()}_API_KEY")

                        text_to_speech = TextToSpeech(model=tts_model, api_key=api_key)
                        # Convert text to speech using the specified model
                        pdf_name = path.split('/')[-1]
                        audio_file_name = pdf_name[:-4] + '.mp3'
                        random_filename = f"podcast_{uuid.uuid4().hex}.mp3"
                        audio_file = os.path.join(
                            output_path, audio_file_name
                        )
                        text_to_speech.convert_to_speech(qa_content, audio_file)
                        logger.info(f"Podcast generated successfully using {tts_model} TTS model")
                        logger.info(f'Time taken: {time.time()-st} secs')
                        return audio_file
                    else:
                        logger.info(f"Transcript generated successfully: {transcript_filepath}")
                        return transcript_filepath

                            
            else:
                return

            

        if generate_audio:
            api_key = None
            # edge does not require an API key
            if tts_model != "edge":
                api_key = getattr(config, f"{tts_model.upper()}_API_KEY")

            text_to_speech = TextToSpeech(model=tts_model, api_key=api_key)
            # Convert text to speech using the specified model
            random_filename = f"podcast_{uuid.uuid4().hex}.mp3"
            audio_file = os.path.join(
                config.get("output_directories")["audio"], random_filename
            )
            text_to_speech.convert_to_speech(qa_content, audio_file)
            logger.info(f"Podcast generated successfully using {tts_model} TTS model")
            logger.info(f'Time taken: {time.time()-st} secs')
            return audio_file
        else:
            logger.info(f"Transcript generated successfully: {transcript_filepath}")
            return transcript_filepath

    except Exception as e:
        logger.error(f"An error occurred in the process_content function: {str(e)}")
        raise

def process_files_wrapper(args):
    return process_content(*args)

@app.command()
def main(
    num_processes: int = typer.Option(None, "--num_processes", "-n", help="Number of parallel processes"),
    path: str = typer.Option(None, "--path", "-p", help="Director to process"),
    output_path: str = typer.Option(None, '--op_path', '-op', help="output_directory"),
    file: typer.FileText = typer.Option(
        None, "--file", "-f", help="File containing URLs, one per line"
    ),
    transcript: typer.FileText = typer.Option(
        None, "--transcript", "-t", help="Path to a transcript file"
    ),
    tts_model: str = typer.Option(
        None,
        "--tts-model",
        "-tts",
        help="TTS model to use (openai, elevenlabs or edge)",
    ),
    transcript_only: bool = typer.Option(
        False, "--transcript-only", help="Generate only a transcript without audio"
    ),
    conversation_config_path: str = typer.Option(
        None,
        "--conversation-config",
        "-cc",
        help="Path to custom conversation configuration YAML file",
    ),
    image_paths: List[str] = typer.Option(
        None, "--image", "-i", help="Paths to image files to process"
    ),
    is_local: bool = typer.Option(
        False,
        "--local",
        "-l",
        help="Use a local LLM instead of a remote one (http://localhost:8080)",
    ),
):
    """
    Generate a podcast or transcript from a list of URLs, a file containing URLs, a transcript file, or image files.
    """
    try:
        config = load_config()
        main_config = config.get("main", {})

        conversation_config = None
        # Load conversation config if provided
        if conversation_config_path:
            with open(conversation_config_path, "r") as f:
                conversation_config: Dict[str, Any] | None = yaml.safe_load(f)
            
                
                
        # Use default TTS model from conversation config if not specified
        if tts_model is None:
            tts_config = load_conversation_config().get('text_to_speech', {})
            tts_model = tts_config.get('default_tts_model', 'openai')
            
        if transcript:
            if image_paths:
                logger.warning("Image paths are ignored when using a transcript file.")
            final_output = process_content(
                transcript_file=transcript.name,
                tts_model=tts_model,
                generate_audio=not transcript_only,
                conversation_config=conversation_config,
                config=config,
                is_local=is_local,
            )
        else:
            all_files = [os.path.join(path, file) for file in os.listdir(path)]
    
            if not all_files:
                print("No files found in the directory.")
                return
            
            # Determine the chunk size (roughly equal-sized chunks for each process)
            chunk_size = len(all_files) // num_processes

            # Split the files into chunks
            file_chunks = chunkify(all_files, chunk_size)
            logger.info(f'File chunks: {file_chunks}')
            arguments = [(chunk, None, output_path, tts_model, True, config, conversation_config, image_paths, is_local) for chunk in file_chunks]

            # Create a multiprocessing Pool
            with Pool(processes=num_processes) as pool:
                # Map each chunk to the process_files function to run in parallel
                pool.map(process_files_wrapper, arguments)

        if transcript_only:
            typer.echo(f"Transcript generated successfully: {final_output}")
        else:
            typer.echo(
                f"Podcast generated successfully using {tts_model} TTS model: {final_output}"
            )

    except Exception as e:
        typer.echo(f"An error occurred: {str(e)}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()


def generate_podcast(
    paths: Optional[List[str]] = None,
    url_file: Optional[str] = None,
    transcript_file: Optional[str] = None,
    tts_model: Optional[str] = None,
    transcript_only: bool = False,
    config: Optional[Dict[str, Any]] = None,
    conversation_config: Optional[Dict[str, Any]] = None,
    image_paths: Optional[List[str]] = None,
    is_local: bool = False,
) -> Optional[str]:
    """
    Generate a podcast or transcript from a list of URLs, a file containing URLs, a transcript file, or image files.

    Args:
        urls (Optional[List[str]]): List of URLs to process.
        url_file (Optional[str]): Path to a file containing URLs, one per line.
        transcript_file (Optional[str]): Path to a transcript file.
        tts_model (Optional[str]): TTS model to use ('openai', 'elevenlabs' or 'edge').
        transcript_only (bool): Generate only a transcript without audio. Defaults to False.
        config (Optional[Dict[str, Any]]): User-provided configuration dictionary.
        conversation_config (Optional[Dict[str, Any]]): User-provided conversation configuration dictionary.
        image_paths (Optional[List[str]]): List of image file paths to process.
        is_local (bool): Whether to use a local LLM. Defaults to False.

    Returns:
        Optional[str]: Path to the final podcast audio file, or None if only generating a transcript.

    Example:
        >>> from podcastfy.client import generate_podcast
        >>> result = generate_podcast(
        ...     image_paths=['/path/to/image1.jpg', '/path/to/image2.png'],
        ...     tts_model='elevenlabs',
        ...     config={
        ...         'main': {
        ...             'default_tts_model': 'elevenlabs'
        ...         },
        ...         'output_directories': {
        ...             'audio': '/custom/path/to/audio',
        ...             'transcripts': '/custom/path/to/transcripts'
        ...         }
        ...     },
        ...     conversation_config={
        ...         'word_count': 150,
        ...         'conversation_style': ['informal', 'friendly'],
        ...         'podcast_name': 'My Custom Podcast'
        ...     },
        ...     is_local=True
        ... )
    """
    try:
        # Load default config
        default_config = load_config()

        # Update config if provided
        if config:
            if isinstance(config, dict):
                # Create a deep copy of the default config
                updated_config = copy.deepcopy(default_config)
                # Update the copy with user-provided values
                updated_config.configure(**config)
                default_config = updated_config
            elif isinstance(config, Config):
                # If it's already a Config object, use it directly
                default_config = config
            else:
                raise ValueError(
                    "Config must be either a dictionary or a Config object"
                )

        main_config = default_config.config.get("main", {})

        # Use provided tts_model if specified, otherwise use the one from config
        if tts_model is None:
            tts_model = main_config.get("default_tts_model", "openai")

        if transcript_file:
            if image_paths:
                logger.warning("Image paths are ignored when using a transcript file.")
            return process_content(
                transcript_file=transcript_file,
                tts_model=tts_model,
                generate_audio=not transcript_only,
                config=default_config,
                conversation_config=conversation_config,
                is_local=is_local,
            )
        else:
            paths_list = paths or []
            if url_file:
                with open(url_file, "r") as file:
                    paths_list.extend([line.strip() for line in file if line.strip()])

            if not paths_list and not image_paths:
                raise ValueError(
                    "No input provided. Please provide either 'urls', 'url_file', 'transcript_file', or 'image_paths'."
                )

            return process_content(
                paths_list=paths_list,
                tts_model=tts_model,
                generate_audio=not transcript_only,
                config=default_config,
                conversation_config=conversation_config,
                image_paths=image_paths,
                is_local=is_local,
            )

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise
