## VARIABLES
SUITE?=unit# unit,integration,k8s
EXTRAS?=pipelines# see sdk's setup.py for all options

sdk.install:
	@echo "::group::Installing sdk with $(EXTRAS) extras"
	pip install ./sdk[$(EXTRAS)]
	@echo "::endgroup::"

sdk.test:
	@echo "::group::Running sdk $(SUITE) test suite"
	@echo "::group:: Commit: $(COMMIT)"
	pytest ./sdk/test/$(SUITE)/ -vv -rfEs
	@echo "::endgroup::"