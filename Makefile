SHELL := /bin/bash
.DEFAULT_GOAL := help

PROJECT     ?= paper-to-aws-routing
AWS_PROFILE ?= personal
AUTH ?= iam
AWS_REGION  ?= us-east-1
ARM         ?= always_strong
PREFIX      ?= routingstudy$(shell echo $(ARM) | tr -d '_')
STACK       ?= $(PREFIX)
ARTIFACTS   ?= $(PREFIX)-artifacts-$(shell aws sts get-caller-identity --query Account --output text 2>/dev/null)
STRONG      ?= anthropic.claude-sonnet-5
WEAK        ?= anthropic.claude-haiku-4-5
THRESHOLD   ?= 0.5

export AWS_PROFILE AWS_REGION

help: ## Show available targets
	@grep -hE '^[a-z%-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

bucket: ## Create the S3 bucket used for Lambda packaging (idempotent)
	@aws s3api head-bucket --bucket $(ARTIFACTS) 2>/dev/null || { \
		aws s3 mb s3://$(ARTIFACTS) && \
		aws s3api put-bucket-tagging --bucket $(ARTIFACTS) \
			--tagging 'TagSet=[{Key=Project,Value=$(PROJECT)}]'; }

up: bucket ## Deploy one arm: make up ARM=length
	aws cloudformation package \
		--template-file infra/gateway.yaml \
		--s3-bucket $(ARTIFACTS) \
		--output-template-file .packaged.yaml
	aws cloudformation deploy \
		--template-file .packaged.yaml \
		--stack-name $(STACK) \
		--capabilities CAPABILITY_IAM \
		--tags Project=$(PROJECT) Arm=$(ARM) ManagedBy=cloudformation \
		--parameter-overrides \
			StackPrefix=$(PREFIX) \
			RouterStrategy=$(ARM) \
			StrongModel=$(STRONG) \
			WeakModel=$(WEAK) \
			RouterThreshold=$(THRESHOLD)
	@$(MAKE) --no-print-directory outputs

outputs: ## Print stack outputs
	@aws cloudformation describe-stacks --stack-name $(STACK) \
		--query 'Stacks[0].Outputs[].[OutputKey,OutputValue]' --output table

inventory: ## List every AWS resource this study created, with teardown commands
	python3 runner/inventory.py

prices: ## Refresh experiments/prices.json from the Marketplace offer API
	python3 runner/fetch_prices.py

smoke: ## One request through the gateway, to prove the path works
	python3 runner/smoke.py --stack $(STACK)

down: ## Delete this arm's stack
	aws cloudformation delete-stack --stack-name $(STACK)
	aws cloudformation wait stack-delete-complete --stack-name $(STACK)
	@echo "deleted $(STACK)"

down-all: ## Delete every arm's stack and the artifact bucket
	@for arm in passthrough always_strong always_weak length routellm; do \
		$(MAKE) --no-print-directory down ARM=$$arm 2>/dev/null || true; \
	done
	-aws s3 rb s3://$(ARTIFACTS) --force

.PHONY: help bucket up outputs inventory prices smoke down down-all

# ---------- inference-time customization pipeline (infra/pipeline.yaml) ----------
PIPE_STACK    ?= gwpipeline
INTERCEPTOR   ?= true
BACKEND       ?= dynamodb
PIPE_ARTIFACT ?= $(PIPE_STACK)-artifacts-$(shell aws sts get-caller-identity --query Account --output text 2>/dev/null)

pipeline-build: ## Assemble the interceptor package (gateway/ + router scorer and weights)
	rm -rf .build/pipeline && mkdir -p .build/pipeline
	cp -R gateway .build/pipeline/gateway
	cp router/routellm_scorer.py .build/pipeline/gateway/
	cp -R router/artifacts .build/pipeline/gateway/artifacts
	cp experiments/prices.json .build/pipeline/gateway/prices.json
	python3 -m pip install -q --target .build/pipeline --upgrade redis   # ElastiCache (Valkey) client
	# The runtime's bundled boto3 predates DynamoDB SearchVectors; ship our own.
	python3 -m pip install -q --target .build/pipeline --upgrade 'boto3>=1.40.100'
	# JWT verification. cryptography ships compiled wheels, so the target must be
	# named explicitly: the function is arm64 on Python 3.13, and a wheel built
	# for the build machine -- or for the wrong architecture -- imports fine here
	# and fails in the function.
	python3 -m pip install -q --target .build/pipeline --upgrade \
		--platform manylinux2014_aarch64 --implementation cp --python-version 3.13 \
		--only-binary=:all: 'PyJWT>=2.8' 'cryptography>=42'
	find .build/pipeline \( -name __pycache__ -o -name "*.dist-info" -o -name "tests" \) -prune -exec rm -rf {} +

pipeline-up: pipeline-build ## Deploy the plugin-pipeline gateway
	@aws s3api head-bucket --bucket $(PIPE_ARTIFACT) 2>/dev/null || { \
		aws s3 mb s3://$(PIPE_ARTIFACT) && aws s3api put-bucket-tagging --bucket $(PIPE_ARTIFACT) \
			--tagging 'TagSet=[{Key=Project,Value=$(PROJECT)}]'; }
	aws cloudformation package --template-file infra/pipeline.yaml \
		--s3-bucket $(PIPE_ARTIFACT) --output-template-file .packaged-pipeline.yaml
	aws cloudformation deploy --template-file .packaged-pipeline.yaml --stack-name $(PIPE_STACK) \
		--capabilities CAPABILITY_IAM --tags Project=$(PROJECT) Component=pipeline ManagedBy=cloudformation \
		--parameter-overrides StackPrefix=$(PIPE_STACK) InterceptorEnabled=$(INTERCEPTOR) CacheBackend=$(BACKEND) AuthMode=$(AUTH)
	@aws cloudformation describe-stacks --stack-name $(PIPE_STACK) \
		--query 'Stacks[0].Outputs[].[OutputKey,OutputValue]' --output table

pipeline-bench: ## Measure each layer's latency through the live gateway (INTERCEPTOR=false stack for "none")
	for c in empty no_guardrail full; do python3 runner/pipeline_bench.py --stack $(PIPE_STACK) --condition $$c; done
	python3 analysis/pipeline_overhead.py

pipeline-down: ## Delete the pipeline stack and its artifact bucket
	aws cloudformation delete-stack --stack-name $(PIPE_STACK)
	aws cloudformation wait stack-delete-complete --stack-name $(PIPE_STACK)
	-aws s3 rb s3://$(PIPE_ARTIFACT) --force

.PHONY: pipeline-build pipeline-up pipeline-bench pipeline-down
