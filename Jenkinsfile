@Library('jenkins-shared-libraries') _

if (env.BRANCH_NAME != null && ! env.BRANCH_NAME.matches("^(master).*")) {
  jobManagement.abortPrevious()
}

def publish_logs(prefix) {
    sh """#!/bin/bash
      set -eux
      rm -rf reports
      mkdir -p reports/${prefix}
      find bazel-out/k8-fastbuild/testlogs/ -name 'test.xml' -exec cp --parents {} ${env.WORKSPACE}/reports/${prefix} \\;
    """
}

pipeline{
  agent {
    kubernetes(
      jnlp.nuplan_devkit(
        name: 'nuplan-devkit-tests',
        tag: "v1.0.3-ubuntu",
        cpu: 8, maxcpu: 8,
        memory: "32G", maxmemory: "64G", yaml: """spec:
  containers:
  - name: builder
    volumeMounts:
      - mountPath: /data
        name: nudeep-ci
        subPath: data
      - mountPath: /dev/shm
        name: dshm
  volumes:
  - name: nudeep-ci
    persistentVolumeClaim:
      claimName: nudeep-ci
  - name: dshm
    emptyDir:
      medium: Memory
"""))
  }

  options {
    timestamps()
    buildDiscarder(logRotator(numToKeepStr:'10'))
  }

  environment {
    BAZEL_CMD  = "bazel --batch"
    BAZEL_OPTS = "--local_cpu_resources=8 --jobs=8 --remote_cache=http://bazel-cache.ci.motional.com:80 --remote_upload_local_results=true"

    NUPLAN_DATA_ROOT         = "/data/sets/nuplan"
    NUPLAN_DB_FILES          = "/data/sets/nuplan/nuplan-v1.1/splits/mini"
    NUPLAN_MAPS_ROOT         = "/data/sets/nuplan/maps"
    NUPLAN_MAP_VERSION       = "nuplan-maps-v1.0"
    NUPLAN_EXP_ROOT          = "/tmp/exp/nuplan"
    NUPLAN_HYDRA_CONFIG_PATH = "config"
  }

  stages {
    stage('Build') {
      steps {
        container('builder') {
          sh """#!/bin/bash -eu
            ${env.BAZEL_CMD} build \
              ${env.BAZEL_OPTS} \
              //...
          """
        }
      }
    }
    stage('Requirements') {
      steps {
        container('builder') {
          sh """#!/bin/bash -eu
            pip3 install -r requirements_torch.txt --index-url=${env.PIP_INDEX_URL_INTERNAL}
            pip3 install -r requirements.txt --index-url=${env.PIP_INDEX_URL_INTERNAL}
            pip3 install tox
          """
        }
      }
    }
    stage('Lint') {
      failFast false
      parallel {
        stage('Bazel') {
          steps {
            container('builder') {
              sh """#!/bin/bash -eu
                ${env.BAZEL_CMD} run :buildifier_test
              """
            }
          }
          post {
            failure {
              println("Consider running 'bazel run :buildifier'")
            }
          }
        }
        stage('Yaml') {
          steps {
            container('builder') {
              sh """#!/bin/bash -eu
                tox -e yamllint -- .
              """
            }
          }
          post {
            failure {
              println("Consider running 'tox -e  yamlformat -- .'")
            }
          }
        }
        stage('Python') {
          steps {
            catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE', message: "Consider running 'tox -e format -- .'") {
              container('builder') {
                sh """#!/bin/bash -eu
                  tox -e lint -- .
                """
              }
            }
          }
          post {
            failure {
              println("Consider running 'tox -e format -- .'")
            }
          }
        }
      }
    }
    stage('Test') {
      steps {
        container('builder') {
          sh """#!/bin/bash -eux
            ${env.BAZEL_CMD} test \
              ${env.BAZEL_OPTS} \
              --action_env=REQUIREMENTS_SHA="\$(sha256sum requirements.txt)" \
              --test_tag_filters=-gpu \
              //...
          """
          publish_logs('gpu_tests')
          junit testResults: 'reports/**/test.xml', allowEmptyResults: true
        }
      }
    }
    stage('Sonarqube') {
      steps {
        container('builder') {
          script {
            sh script: """#!/usr/bin/env bash
              env
              rm -rf bazel-* # remove tree with symlinks, as it may cause sonar-scanner to slow down ot stuck
            """
            sonarQube.scanner('builder', """ \
            """, true)
          }
        }
      }
    }
  }
}
