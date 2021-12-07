@Library('jenkins-shared-libraries') _

if (env.BRANCH_NAME != null && ! env.BRANCH_NAME.matches("^(master).*")) {
  jobManagement.abortPrevious()
}

pipeline{
  agent {
    kubernetes(
      jnlp.builder(
        name: 'nuplan-devkit-tests',
        builder_image: "233885420847.dkr.ecr.us-east-1.amazonaws.com/nuplan-devkit:v1.0.1",
        cpu: 1, maxcpu: 2,
        memory: "1G", maxmemory: "5G"))
  }

  options {
    timestamps()
    buildDiscarder(logRotator(numToKeepStr:'10'))
  }

  environment {
    BAZEL_CMD = "bazel --batch"
    BAZEL_OPTS = "--remote_upload_local_results=true"
  }

  stages{
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
    stage('Test') {
      steps {
        container('builder') {
          sh """#!/bin/bash -eu
            ${env.BAZEL_CMD} test \
              ${env.BAZEL_OPTS} \
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
            env.TESTS = sh (
              script: "find . -name 'tests' -not -path './.bazel/*' -type d -printf %p,",
              returnStdout: true
            ).trim()
            sonarQube.scanner('builder', """ \
              -Dsonar.tests=${env.TESTS} \
            """, true)
          }
        }
      }
    }
  }
}
