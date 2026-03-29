# Streaming

Token-level streaming components.

## TokenEvent

::: deepagent_temporal.streaming.TokenEvent
    options:
        show_source: true

## TokenCapturingHandler

::: deepagent_temporal.streaming.TokenCapturingHandler
    options:
        show_source: true
        members:
            - on_llm_new_token
            - on_llm_end
            - on_chat_model_start
            - token_count

## RedisStreamBackend

::: deepagent_temporal.streaming.RedisStreamBackend
    options:
        show_source: true
        members:
            - publish
            - publish_complete
            - subscribe
            - cleanup
            - close

## StreamingNodeWrapper

::: deepagent_temporal.activity.StreamingNodeWrapper
    options:
        show_source: true
        members:
            - ainvoke

## Helper Functions

::: deepagent_temporal.streaming.create_token_handler

::: deepagent_temporal.activity.wrap_graph_for_streaming

::: deepagent_temporal.worker.create_streaming_worker
