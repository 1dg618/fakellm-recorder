"""fakellm-recorder: record real OpenAI/Anthropic traffic, emit fakellm.yaml rules."""

__version__ = "0.1.0"

# fakellm config schema version this emitter targets. fakellm is beta and
# single-author, so we stamp emitted files and gate on this.
TARGET_FAKELLM_CONFIG_VERSION = 1
