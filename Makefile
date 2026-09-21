SHELL := /bin/bash
.DEFAULT_GOAL := help

PROJECT     ?= paper-to-aws-routing
AWS_PROFILE ?= personal
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
