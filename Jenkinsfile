@Library('jenkins-shared-libraries') _

if (env.BRANCH_NAME != null && ! env.BRANCH_NAME.matches("^(master).*")) {
  jobManagement.abortPrevious()
}

pipeline{
  agent {
    kubernetes(
      jnlp.nuplan_devkit(
        name: 'nuplan-devkit-tests',
        tag: "v1.0.2",
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
    BAZEL_CMD        = "bazel --batch"
    BAZEL_OPTS       = "--local_cpu_resources=8 --jobs=8 --remote_cache=http://bazel-cache.ci.motional.com:80 --remote_upload_local_results=true"
    NUPLAN_DATA_ROOT = "/data/sets/nuplan"
    NUPLAN_EXP_ROOT  = "/tmp/exp/nuplan"
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
            pip3 install -r requirements.txt \
              --index-url=${env.PIP_INDEX_URL_INTERNAL}
          """
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
